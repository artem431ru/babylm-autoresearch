#!/usr/bin/env python3
"""Validate frozen short-search winners at 2048 steps on new seeds (no API).

Place beside train.py and crs_repeat_comparison.json.
Preview: python validate_long.py --dry-run
Run or continue: python validate_long.py
Output: long_validation/results.json
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
SEEDS = (46, 47, 48)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")
    temporary.replace(path)


def make_plan(comparison_path, data):
    source = read_json(comparison_path)
    if not isinstance(source, list) or len(source) != 9:
        raise ValueError("Expected the complete nine-search crs_repeat_comparison.json.")
    expected_pairs = {(seed, mode) for seed in (43,44,45) for mode in ("random","agent","no-history")}
    pairs = {(r['summary']['training_seed'], r['summary']['mode']) for r in source}
    if pairs != expected_pairs or any(r['summary']['status'] != 'completed' for r in source):
        raise ValueError("Expected completed searches for all three methods on seeds 43,44,45.")
    reference = source[0]['plan']
    train_hash = digest(ROOT / 'train.py')
    if train_hash != reference['source_sha256']['train.py']:
        raise ValueError("train.py changed since the search. Restore it before validating the selected recipes.")
    data_hashes = {name:digest(data / name) for name in reference['data_sha256']}
    if data_hashes != reference['data_sha256']:
        raise ValueError("Prepared data changed since the search.")
    comparable = ('search_space','trials','data_sha256','source_sha256','model','api_endpoint','reasoning_effort')
    for record in source:
        if any(record['plan'][k] != reference[k] for k in comparable):
            raise ValueError("Search methods have incompatible data, code or API settings.")
        config = dict(record['plan']['base_config'])
        config.pop('seed')
        ref_config = dict(reference['base_config'])
        ref_config.pop('seed')
        if config != ref_config:
            raise ValueError("Search budgets or base configurations differ.")

    recipes = {'baseline': deepcopy(reference['base_config'])}
    selection = {'baseline': dict(source_search=source[0]['name'], selection='common baseline')}
    for mode in ('random','no-history','agent'):
        winner = min((r for r in source if r['summary']['mode'] == mode),
                     key=lambda r: r['summary']['best_final_dev_loss'])
        recipes[mode] = deepcopy(winner['summary']['best_config'])
        selection[mode] = dict(source_search=winner['name'], source_trial=winner['summary']['best_trial'],
            source_loss=winner['summary']['best_final_dev_loss'],
            selection='lowest final dev loss among this method\'s three short-search winners')
    jobs = []
    methods = list(recipes)
    for index, seed in enumerate(SEEDS):
        # Rotate order across seeds; wall-clock speed is not the primary metric.
        order = methods[index:] + methods[:index]
        for mode in order:
            config = dict(recipes[mode], seed=seed, max_steps=2048, eval_interval=256, log_interval=256)
            jobs.append(dict(name=f"{mode}_s{seed}", method=mode, config=config))
    return dict(schema_version=1, comparison_sha256=digest(comparison_path),
        train_sha256=train_hash, launcher_sha256=digest(Path(__file__).resolve()),
        data_directory=str(data), data_sha256=data_hashes, seeds=list(SEEDS), jobs=jobs,
        selected_recipes=selection, primary_metric='last_measured_dev_loss',
        note='Recipe transfer to longer training on new training seeds. Not a new search; same internal dev windows remain in use.')


def validate_result(folder, job, plan):
    path = folder / 'results.json'
    if not path.is_file():
        raise ValueError(f"Incomplete training: {folder}. Move that incomplete folder aside, then rerun.")
    r = read_json(path)
    config = job['config']
    if (r.get('status') != 'completed' or r.get('config') != config
            or r.get('script_sha256') != plan['train_sha256']
            or r.get('completed_steps') != config['max_steps']
            or r.get('tokens_seen') != config['max_steps']*config['batch_size']*config['block_size']):
        raise ValueError(f"Incomplete or incompatible result: {folder}")
    if not isinstance(r.get('last_measured_dev_loss'), (int,float)) or not math.isfinite(r['last_measured_dev_loss']):
        raise ValueError(f"Invalid loss in {folder}")
    return r


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison', type=Path, default=ROOT / 'crs_repeat_comparison.json')
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data' / 'processed')
    parser.add_argument('--out-dir', type=Path, default=ROOT / 'long_validation')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    plan = make_plan(args.comparison.resolve(), args.data_dir.resolve())
    if args.dry_run:
        print(json.dumps(dict(trainings=len(plan['jobs']), steps=2048, seeds=list(SEEDS), api_requests=0,
            tokens_per_training=8*128*2048, selected_recipes=plan['selected_recipes'],
            configs={j['method']:j['config'] for j in plan['jobs'] if j['config']['seed']==46}),
            ensure_ascii=False,indent=2))
        return 0
    output = args.out_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    plan_path = output / 'plan.json'
    if plan_path.exists():
        if read_json(plan_path) != plan:
            raise ValueError('Validation plan, source or data changed. Keep the original setup or use another --out-dir.')
    else:
        if any(output.iterdir()):
            raise ValueError('Output directory is nonempty without a saved validation plan. Choose another --out-dir.')
        write_json(plan_path, plan)
    rows = []
    for i, job in enumerate(plan['jobs'], 1):
        folder = output / job['name']
        config_path = output / f"{job['name']}.config.json"
        print(f"\n[{i}/{len(plan['jobs'])}] {job['name']} | 2048 steps | no API", flush=True)
        if not folder.exists():
            write_json(config_path, job['config'])
            command = [sys.executable, '-u', str(ROOT / 'train.py'), '--config', str(config_path),
                       '--data-dir', plan['data_directory'], '--out-dir', str(folder)]
            code = subprocess.run(command, check=False).returncode
            if code:
                raise ValueError(f"Training failed ({code}) for {job['name']}. Completed results are retained.")
        else:
            print('Reading saved result; training is not repeated.', flush=True)
        result = validate_result(folder, job, plan)
        rows.append(dict(method=job['method'], seed=job['config']['seed'], result=result))
        write_json(output / 'results.json', dict(status='completed' if len(rows)==len(plan['jobs']) else 'running',
            plan=plan, completed_trainings=len(rows), api_requests=0, runs=rows))
    print(f"\nCOMPLETED: {output / 'results.json'}", flush=True)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('Stopped. Rerun the same command to continue completed results.', file=sys.stderr)
        raise SystemExit(130)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        raise SystemExit(1)
