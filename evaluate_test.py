#!/usr/bin/env python3
"""Final internal test evaluation of 12 saved last.pt checkpoints; no training/API.

Place beside train.py and the long_validation folder.
Preview: python evaluate_test.py --dry-run
Evaluate/continue: python evaluate_test.py
Output: test_evaluation/results.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parent


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(path):
    checksum = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


def write_json(path, payload):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def prepare_plan(validation_dir, data_dir, batch_size, threads):
    source_path = validation_dir / 'results.json'
    source = read_json(source_path)
    previous = source['plan']
    if source.get('status') != 'completed' or source.get('completed_trainings') != 12:
        raise ValueError('Finish all 12 long-validation trainings before opening the test set.')
    train_hash = digest(ROOT / 'train.py')
    if train_hash != previous['train_sha256']:
        raise ValueError('train.py changed. Restore the training version before loading checkpoints.')
    manifest_path = data_dir / 'manifest.json'
    manifest = read_json(manifest_path)
    if digest(manifest_path) != previous['data_sha256']['manifest.json']:
        raise ValueError('manifest.json differs from the manifest used in training.')
    tokenizer_hash = digest(data_dir / 'tokenizer.json')
    if tokenizer_hash != previous['data_sha256']['tokenizer.json']:
        raise ValueError('Tokenizer changed since training.')
    info = manifest['splits']['test']
    test_path = data_dir / 'test.bin'
    if (manifest['binary']['dtype'] != '<u2' or test_path.stat().st_size != 2*info['tokens']
            or digest(test_path) != info['sha256']):
        raise ValueError('test.bin is missing, changed or incompatible with its manifest.')
    if info['tokens'] < 2:
        raise ValueError('Test set is too short.')
    expected = {(seed, method) for seed in (46,47,48) for method in ('baseline','random','no-history','agent')}
    observed = {(row['seed'],row['method']) for row in source['runs']}
    if len(source['runs']) != 12 or observed != expected:
        raise ValueError('Validation must contain the four frozen recipes on seeds 46,47,48.')
    previous_jobs = {(job['config']['seed'],job['method']):job for job in previous['jobs']}
    jobs = []
    for row in source['runs']:
        seed, method, result = row['seed'], row['method'], row['result']
        config = result['config']
        selected = previous_jobs[(seed,method)]
        if (result['status'] != 'completed' or config != selected['config']
                or result['script_sha256'] != train_hash or result['completed_steps'] != 2048
                or result['tokens_seen'] != config['max_steps']*config['batch_size']*config['block_size']):
            raise ValueError(f'Incomplete or mismatched training: {method}, seed {seed}.')
        name = selected['name']
        if name != f'{method}_s{seed}':
            raise ValueError('Unexpected checkpoint directory name in validation plan.')
        checkpoint = validation_dir / name / 'last.pt'
        if not checkpoint.is_file():
            raise ValueError(f'Missing saved checkpoint: {checkpoint}. Keep the long_validation folders.')
        jobs.append(dict(name=name, method=method, seed=seed, checkpoint=str(checkpoint),
                         checkpoint_sha256=digest(checkpoint), config=config,
                         previous_dev_loss=result['last_measured_dev_loss']))
    lengths = {j['config']['block_size'] for j in jobs}
    if len(lengths) != 1:
        raise ValueError('All models must use the same context length for this test protocol.')
    return dict(schema_version=1, source_validation_sha256=digest(source_path),
        evaluator_sha256=digest(Path(__file__).resolve()), train_sha256=train_hash,
        manifest_sha256=digest(manifest_path), tokenizer_sha256=tokenizer_hash,
        validation_directory=str(validation_dir), test_path=str(test_path), test_sha256=info['sha256'],
        test_tokens=info['tokens'], scored_tokens=info['tokens']-1,
        vocab_size=manifest['tokenizer']['vocab_size'], block_size=next(iter(lengths)),
        batch_size=batch_size, threads=threads, checkpoint_selection='last.pt fixed before test evaluation',
        protocol='Sequential non-overlapping target blocks. Reset context at each block; include final partial block. Score every token except the first once.',
        primary_metric='mean next-token cross-entropy, nats per token',
        note='Internal held-out line-group split, not a document-disjoint split or official BabyLM benchmark. No parameter tuning on these test results.',
        jobs=jobs)


def load_model(job, plan):
    import torch
    from train import TinyGPT
    checkpoint = torch.load(job['checkpoint'], map_location='cpu', weights_only=True)
    config = job['config']
    if (checkpoint.get('format_version') != 1 or checkpoint.get('training_config') != config
            or checkpoint.get('step') != config['max_steps']
            or checkpoint.get('tokens_seen') != config['max_steps']*config['batch_size']*config['block_size']):
        raise ValueError(f"Checkpoint metadata differs from the frozen recipe: {job['name']}")
    identity = checkpoint['data_identity']
    if (identity['manifest_sha256'] != plan['manifest_sha256']
            or identity['tokenizer_sha256'] != plan['tokenizer_sha256']):
        raise ValueError(f"Checkpoint data identity mismatch: {job['name']}")
    model_config = dict(vocab_size=plan['vocab_size'], **{k:config[k] for k in
                        ('block_size','n_layer','n_head','n_embd','dropout')})
    if checkpoint['model_config'] != model_config:
        raise ValueError(f"Checkpoint architecture mismatch: {job['name']}")
    model = TinyGPT(**model_config)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.eval()
    return model


def evaluate_full(model, stream, block_size, batch_size):
    import numpy as np
    import torch
    from train import batch_at
    if len(stream) < 2:
        raise ValueError('At least two tokens are required.')
    scored = 0
    total_nll = 0.0
    target_tokens = len(stream)-1
    complete_blocks, tail = divmod(target_tokens,block_size)
    model.eval()
    with torch.inference_mode():
        for first in range(0,complete_blocks,batch_size):
            starts = np.arange(first,min(first+batch_size,complete_blocks),dtype=np.int64)*block_size
            x,y = batch_at(stream,starts,block_size,torch.device('cpu'))
            _,loss = model(x,y)
            total_nll += float(loss.item())*y.numel()
            scored += y.numel()
        if tail:
            x,y = batch_at(stream,np.asarray([complete_blocks*block_size]),tail,torch.device('cpu'))
            _,loss = model(x,y)
            total_nll += float(loss.item())*y.numel()
            scored += y.numel()
    if scored != target_tokens:
        raise ValueError('Test token coverage mismatch.')
    loss = total_nll/scored
    if not math.isfinite(loss):
        raise ValueError('Non-finite test loss.')
    return dict(test_loss=loss, perplexity=math.exp(loss), scored_tokens=scored,
                evaluated_blocks=complete_blocks+bool(tail))


def summarize(rows):
    summary = {}
    for method in ('baseline','random','no-history','agent'):
        selected = [r for r in rows if r['method']==method]
        values = [r['test_loss'] for r in selected]
        if values:
            summary[method] = dict(completed_models=len(values), mean_test_loss=statistics.mean(values),
                std_test_loss=statistics.stdev(values) if len(values)>1 else None,
                mean_perplexity=statistics.mean(r['perplexity'] for r in selected))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validation-dir', type=Path, default=ROOT/'long_validation')
    parser.add_argument('--data-dir', type=Path, default=ROOT/'data'/'processed')
    parser.add_argument('--out-dir', type=Path, default=ROOT/'test_evaluation')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--threads', type=int, default=10)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if min(args.batch_size,args.threads) < 1:
        raise ValueError('Batch size and threads must be positive.')
    plan = prepare_plan(args.validation_dir.resolve(),args.data_dir.resolve(),args.batch_size,args.threads)
    if args.dry_run:
        print(json.dumps(dict(saved_models=len(plan['jobs']),trainings=0,api_requests=0,
            test_tokens=plan['test_tokens'],scored_tokens_per_model=plan['scored_tokens'],
            checkpoint_selection=plan['checkpoint_selection'],protocol=plan['protocol'],note=plan['note']),
            ensure_ascii=False,indent=2))
        return 0
    import numpy as np
    import torch
    torch.set_num_threads(args.threads)
    plan['runtime'] = dict(torch=str(torch.__version__),numpy=np.__version__,python=sys.version.split()[0])
    stream = np.memmap(plan['test_path'],dtype='<u2',mode='r')
    if int(stream.max()) >= plan['vocab_size']:
        raise ValueError('Test token IDs do not match the saved tokenizer vocabulary.')
    output = args.out_dir.resolve()
    output.mkdir(parents=True,exist_ok=True)
    plan_path = output/'plan.json'
    if plan_path.exists():
        if read_json(plan_path) != plan:
            raise ValueError('Test plan, files or runtime changed. Restore the original setup or use another --out-dir.')
    else:
        if any(output.iterdir()):
            raise ValueError('Nonempty test output directory without a saved plan. Choose another --out-dir.')
        write_json(plan_path,plan)
    rows = []
    for index,job in enumerate(plan['jobs'],1):
        cache = output/f"{job['name']}.json"
        print(f"[{index}/{len(plan['jobs'])}] {job['name']} | test only | no API",flush=True)
        if cache.exists():
            row = read_json(cache)
            if (row.get('checkpoint_sha256') != job['checkpoint_sha256']
                    or row.get('scored_tokens') != plan['scored_tokens']
                    or row.get('test_sha256') != plan['test_sha256']
                    or row.get('method') != job['method'] or row.get('seed') != job['seed']
                    or not isinstance(row.get('test_loss'),(int,float)) or not math.isfinite(row['test_loss'])):
                raise ValueError(f"Invalid cached test result: {cache}")
        else:
            started = time.perf_counter()
            model = load_model(job,plan)
            measured = evaluate_full(model,stream,plan['block_size'],plan['batch_size'])
            row = dict(method=job['method'],seed=job['seed'],checkpoint_sha256=job['checkpoint_sha256'],
                test_sha256=plan['test_sha256'],previous_dev_loss=job['previous_dev_loss'],
                evaluation_seconds=time.perf_counter()-started,**measured)
            write_json(cache,row)
            del model
        rows.append(row)
        print(f"  test_loss={row['test_loss']:.6f}; perplexity={row['perplexity']:.3f}",flush=True)
        write_json(output/'results.json',dict(status='completed' if len(rows)==len(plan['jobs']) else 'running',
            plan=plan,completed_evaluations=len(rows),trainings=0,api_requests=0,runs=rows,summary=summarize(rows)))
    print(f"COMPLETED: {output/'results.json'}",flush=True)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('Stopped. Completed test evaluations are retained; rerun the same command.',file=sys.stderr)
        raise SystemExit(130)
    except (OSError,ValueError,KeyError,TypeError,RuntimeError) as error:
        print(f'ERROR: {error}',file=sys.stderr)
        raise SystemExit(1)
