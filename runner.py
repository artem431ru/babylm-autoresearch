#!/usr/bin/env python3
"""Compare two learning rates with paired seeds using the existing train.py.

Run: python runner.py
Preview: python runner.py --dry-run
Continue an interrupted suite: python runner.py --resume
Uses Python's standard library; train.py needs the same dependencies as before.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
HYPOTHESIS = "Peak LR 0.001 gives lower final dev loss than 0.0003 at a fixed token budget."
BASE = dict(max_steps=512, batch_size=8, block_size=128, n_layer=2,
            n_head=4, n_embd=128, dropout=0.1, weight_decay=0.1,
            warmup_fraction=0.1, min_lr_ratio=0.1, grad_clip=1.0,
            eval_interval=64, eval_batches=20, eval_batch_size=4,
            eval_seed=12345, log_interval=64, threads=10, device="cpu")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def make_jobs(base):
    jobs = []
    for i, seed in enumerate((42, 43, 44)):
        # Alternate the order to reduce a systematic warm-up/temperature bias.
        rates = (0.0003, 0.001) if i % 2 == 0 else (0.001, 0.0003)
        for lr in rates:
            jobs.append(dict(name=f"lr{str(lr).replace('.', 'p')}_seed{seed}",
                             config=dict(base, learning_rate=lr, seed=seed)))
    return jobs


def validated_result(folder, job, script_hash):
    if not (folder / "results.json").is_file():
        raise ValueError(f"Incomplete run: {folder}. Move this folder aside and use --resume.")
    result = read_json(folder / "results.json")
    config = job["config"]
    budget = config["max_steps"] * config["batch_size"] * config["block_size"]
    if (result.get("status") != "completed" or result.get("config") != config
            or result.get("script_sha256") != script_hash
            or result.get("completed_steps") != config["max_steps"]
            or result.get("tokens_seen") != budget):
        raise ValueError(f"Incomplete or incompatible result: {folder}")
    for key in ("last_measured_dev_loss", "best_dev_loss", "training_seconds",
                "run_elapsed_seconds", "train_tokens_per_second"):
        if not math.isfinite(result[key]):
            raise ValueError(f"Non-finite {key}: {folder}")
    return result


def row_for(job, result, folder):
    return dict(run=job["name"], hypothesis=HYPOTHESIS, status="completed",
                seed=job["config"]["seed"], learning_rate=job["config"]["learning_rate"],
                final_dev_loss=result["last_measured_dev_loss"],
                best_dev_loss=result["best_dev_loss"], tokens_seen=result["tokens_seen"],
                training_seconds=result["training_seconds"],
                run_elapsed_seconds=result["run_elapsed_seconds"],
                train_tokens_per_second=result["train_tokens_per_second"],
                train_script_sha256=result["script_sha256"], run_directory=str(folder))


def write_journal(suite, rows):
    if not rows:
        return
    temporary = suite / "journal.csv.tmp"
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(suite / "journal.csv")


def summarize(suite, rows):
    groups = []
    for lr in (0.0003, 0.001):
        selected = [r for r in rows if r["learning_rate"] == lr]
        losses = [r["final_dev_loss"] for r in selected]
        groups.append(dict(learning_rate=lr, seeds=[r["seed"] for r in selected],
                           mean_final_dev_loss=statistics.mean(losses),
                           sample_std_final_dev_loss=statistics.stdev(losses),
                           mean_training_seconds=statistics.mean(r["training_seconds"] for r in selected)))
    pairs = []
    for seed in (42, 43, 44):
        low = next(r for r in rows if r["seed"] == seed and r["learning_rate"] == 0.0003)
        high = next(r for r in rows if r["seed"] == seed and r["learning_rate"] == 0.001)
        pairs.append(dict(seed=seed, low_lr_final_dev_loss=low["final_dev_loss"],
                          high_lr_final_dev_loss=high["final_dev_loss"],
                          loss_reduction=low["final_dev_loss"] - high["final_dev_loss"]))
    output = dict(hypothesis=HYPOTHESIS, primary_metric="final dev loss (lower is better)",
                  groups=groups, paired_comparisons=pairs,
                  mean_paired_loss_reduction=statistics.mean(p["loss_reduction"] for p in pairs),
                  higher_lr_wins=sum(p["loss_reduction"] > 0 for p in pairs),
                  note="Three paired seeds on fixed internal sampled dev windows. Exploratory evidence; not an official BabyLM score. LR was previously selected using this dev split. Test is not read.")
    write_json(suite / "comparison.json", output)
    print("\nLR       Mean final dev loss   Sample std   Mean training seconds")
    for group in groups:
        print(f"{group['learning_rate']:<8g} {group['mean_final_dev_loss']:<21.5f} "
              f"{group['sample_std_final_dev_loss']:<12.5f} {group['mean_training_seconds']:.2f}")
    print(f"Higher LR wins: {output['higher_lr_wins']}/3 seeds. "
          f"Mean paired loss reduction: {output['mean_paired_loss_reduction']:.5f}")


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="lr_check_01")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--threads", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=512)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.name in ("", ".", "..") or any(c in args.name for c in '/\\:'):
        parser.error("--name must be a folder name.")
    if args.threads < 1 or args.max_steps < 1:
        parser.error("--threads and --max-steps must be positive.")
    train = ROOT / "train.py"
    data = args.data_dir.resolve()
    base = dict(BASE, threads=args.threads, max_steps=args.max_steps,
                eval_interval=max(1, args.max_steps // 8),
                log_interval=max(1, args.max_steps // 8))
    jobs = make_jobs(base)
    plan = dict(schema_version=1, hypothesis=HYPOTHESIS,
                primary_metric="final dev loss", train_script_sha256=digest(train),
                runner_sha256=digest(Path(__file__)), data_directory=str(data),
                data_sha256={name: digest(data / name) for name in
                             ("manifest.json", "tokenizer.json", "train.bin", "dev.bin")},
                jobs=jobs)
    suite = ROOT / "experiments" / args.name
    if args.dry_run:
        for job in jobs:
            print(f"{job['name']}: {job['config']['max_steps']} steps; "
                  f"{base['batch_size'] * base['block_size'] * base['max_steps']:,} tokens")
        print(f"Output: {suite}\nNo training started; no files written.")
        return
    if args.resume:
        if read_json(suite / "plan.json") != plan:
            raise ValueError("Plan, data or code changed. Choose a new --name without --resume.")
    else:
        suite.mkdir(parents=True, exist_ok=False)
        write_json(suite / "plan.json", plan)
    rows = []
    for number, job in enumerate(jobs, 1):
        folder = suite / job["name"]
        config_path = suite / (job["name"] + ".json")
        if folder.exists():
            result = validated_result(folder, job, plan["train_script_sha256"])
            print(f"[{number}/6] Reusing completed {job['name']}", flush=True)
        else:
            write_json(config_path, job["config"])
            command = [sys.executable, "-u", str(train), "--config", str(config_path),
                       "--data-dir", str(data), "--out-dir", str(folder)]
            print(f"\n[{number}/6] {job['name']}", flush=True)
            write_json(suite / "current_run.json", dict(run=job["name"], command=command,
                       started_utc=datetime.now(timezone.utc).isoformat()))
            with (suite / (job["name"] + ".log")).open("w", encoding="utf-8") as log:
                process = subprocess.Popen(command, stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT, text=True,
                                           encoding="utf-8", errors="replace",
                                           env=dict(os.environ, PYTHONIOENCODING="utf-8"))
                try:
                    for line in process.stdout:
                        log.write(line)
                        log.flush()
                        print(line, end="", flush=True)
                    code = process.wait()
                except BaseException:
                    if process.poll() is None:
                        process.terminate()
                        process.wait()
                    raise
            if code:
                raise RuntimeError(f"Training failed ({code}); see {job['name']}.log")
            result = validated_result(folder, job, plan["train_script_sha256"])
        rows.append(row_for(job, result, folder))
        write_journal(suite, rows)
    summarize(suite, rows)
    print(f"\nSaved journal.csv and comparison.json to: {suite}")


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        print("\nStopped. Completed runs are preserved. To retry an incomplete run, move its folder aside, then use --resume.", file=sys.stderr)
        sys.exit(130)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
