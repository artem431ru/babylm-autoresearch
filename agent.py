#!/usr/bin/env python3
"""Bounded BabyLM recipe search: OpenAI agent, no-history ablation, random search.

Requires existing train.py and runner.py beside this file. No new dependencies.
python agent.py --mode random --name random_01
python agent.py --mode agent --name agent_01
python agent.py --mode no-history --name nohist_01
python agent.py --compare random_01 agent_01 nohist_01
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import itertools
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import urllib.error
import urllib.request

from runner import BASE, digest, read_json, validated_result, write_json

ROOT = Path(__file__).resolve().parent
SPACE = {"learning_rate": [0.0005, 0.001, 0.002, 0.003],
         "weight_decay": [0.0, 0.01, 0.1],
         "dropout": [0.0, 0.1, 0.2],
         "warmup_fraction": [0.0, 0.1, 0.2]}
BASELINE = dict(learning_rate=0.001, weight_decay=0.1, dropout=0.1, warmup_fraction=0.1)
INSTRUCTIONS = """You choose training recipes for a small causal language model.
Goal: minimize FINAL cross-entropy on fixed internal dev windows at a fixed token budget.
Each trial starts from scratch, with the same architecture, training seed and sampled data
sequence. A baseline is always provided. Choose exactly one available candidate_id.
Use the supplied evidence when available. Reason about LR, regularization and warmup
in the SHORT training regime. Do not invent measurements, read test data, request tools,
or change the budget. Explain one testable hypothesis and the expected effect concisely
in Russian (at most three sentences each). Output the required JSON object.
"""


def candidates():
    keys = list(SPACE)
    return {f"c{i:03d}": dict(zip(keys, values)) for i, values in enumerate(
            itertools.product(*(SPACE[k] for k in keys)))
            if dict(zip(keys, values)) != BASELINE}


def parse_response(response, remaining):
    if response.get("status") != "completed":
        raise ValueError("OpenAI response was incomplete or failed. See api_response.json.")
    parts = [part for item in response.get("output", []) if item.get("type") == "message"
             for part in item.get("content", [])]
    if any(part.get("type") == "refusal" for part in parts):
        raise ValueError("OpenAI refused the proposal. No training was started.")
    text = "".join(part.get("text", "") for part in parts if part.get("type") == "output_text")
    proposal = json.loads(text)
    if not isinstance(proposal, dict) or set(proposal) != {"candidate_id", "hypothesis", "expected_effect"}:
        raise ValueError("Invalid proposal structure.")
    if not isinstance(proposal["candidate_id"], str) or proposal["candidate_id"] not in remaining:
        raise ValueError("Proposal is not an available candidate; no training started.")
    for key in ("hypothesis", "expected_effect"):
        if not isinstance(proposal[key], str) or not 1 <= len(proposal[key]) <= 2000:
            raise ValueError(f"Invalid {key} in proposal.")
    return proposal


def make_request(model, max_tokens, remaining, context):
    schema = {"type": "object", "properties": {
        "candidate_id": {"type": "string", "enum": list(remaining)},
        "hypothesis": {"type": "string"}, "expected_effect": {"type": "string"}},
        "required": ["candidate_id", "hypothesis", "expected_effect"], "additionalProperties": False}
    return dict(model=model, store=False, max_output_tokens=max_tokens,
                reasoning={"effort": "low"}, instructions=INSTRUCTIONS,
                input=json.dumps(dict(context, available_candidates=remaining), ensure_ascii=False),
                text={"format": dict(type="json_schema", name="experiment_proposal",
                                     strict=True, schema=schema)})


def call_openai(body, key):
    request = urllib.request.Request("https://api.openai.com/v1/responses",
              data=json.dumps(body).encode("utf-8"), method="POST",
              headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    # No automatic retries: each trial makes at most one API request per invocation.
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace").replace(key, "[REDACTED]")[:1000]
        raise RuntimeError(f"OpenAI HTTP {error.code}: {detail}") from None
    except (urllib.error.URLError, TimeoutError) as error:
        raise RuntimeError("OpenAI connection failed. Check connection/proxy and retry with --resume.") from error


def train_once(config, folder, data):
    config_path = folder / "config.json"
    write_json(config_path, config)
    command = [sys.executable, "-u", str(ROOT / "train.py"), "--config", str(config_path),
               "--data-dir", str(data), "--out-dir", str(folder / "training")]
    with (folder / "training.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                  text=True, encoding="utf-8", errors="replace",
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
        raise RuntimeError(f"Training failed ({code}); see {folder / 'training.log'}")


def save_progress(suite, rows, plan):
    write_json(suite / "history.json", rows)
    columns = ["trial", "candidate_id", "hypothesis", "expected_effect", *SPACE,
               "final_dev_loss", "best_so_far_dev_loss", "training_seconds", "tokens_seen"]
    temporary = suite / "journal.csv.tmp"
    best = math.inf
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            best = min(best, row["final_dev_loss"])
            writer.writerow(dict(trial=row["trial"], candidate_id=row["proposal"]["candidate_id"],
                 hypothesis=row["proposal"]["hypothesis"], expected_effect=row["proposal"]["expected_effect"],
                 **{k: row["config"][k] for k in SPACE}, final_dev_loss=row["final_dev_loss"],
                 best_so_far_dev_loss=best, training_seconds=row["training_seconds"],
                 tokens_seen=row["tokens_seen"]))
    temporary.replace(suite / "journal.csv")
    winner = min(rows, key=lambda row: row["final_dev_loss"])
    baseline_loss = rows[0]["final_dev_loss"]
    summary = dict(mode=plan["mode"], status="completed" if len(rows) == plan["trials"] + 1 else "running",
                   training_seed=plan["base_config"]["seed"], completed_trainings=len(rows),
                   baseline_dev_loss=baseline_loss, best_final_dev_loss=winner["final_dev_loss"],
                   loss_reduction=baseline_loss-winner["final_dev_loss"], best_trial=winner["trial"],
                   best_config=winner["config"], training_tokens=sum(r["tokens_seen"] for r in rows),
                   total_training_seconds=sum(r["training_seconds"] for r in rows),
                   total_api_input_tokens=sum((r.get("api_usage") or {}).get("input_tokens", 0) for r in rows),
                   total_api_output_tokens=sum((r.get("api_usage") or {}).get("output_tokens", 0) for r in rows),
                   note="Single-seed adaptive search on internal dev. Re-evaluate winners on new seeds; do not claim official BabyLM performance. API latency/cost is separate from training budget.")
    write_json(suite / "summary.json", summary)
    write_json(suite / "best_config.json", winner["config"])
    return summary


def build_context(base, rows, mode):
    return dict(task="BabyLM short-budget causal LM recipe optimization", fixed_config=base,
                primary_metric="final dev cross-entropy, lower is better",
                baseline=rows[0],
                completed_search_trials=rows[1:] if mode == "agent" else [],
                history_policy="all completed outcomes" if mode == "agent" else
                    "baseline only; candidate list excludes previous choices, but their outcomes are hidden")


def search(args):
    data = args.data_dir.resolve()
    base = dict(BASE, max_steps=args.max_steps, batch_size=8, threads=args.threads,
                seed=args.seed, eval_interval=max(1, args.max_steps // 8),
                log_interval=max(1, args.max_steps // 8), **BASELINE)
    pool = candidates()
    plan = dict(schema_version=1, mode=args.mode, trials=args.trials, model=args.model,
                max_output_tokens=args.max_output_tokens, search_seed=args.search_seed,
                base_config=base, search_space=SPACE, data_directory=str(data),
                data_sha256={name: digest(data / name) for name in
                             ("manifest.json", "tokenizer.json", "train.bin", "dev.bin")},
                source_sha256={name: digest(ROOT / name) for name in ("train.py", "runner.py", "agent.py")})
    if args.dry_run:
        print(json.dumps(dict(mode=args.mode, model=args.model, trainings=args.trials + 1,
              api_requests=0 if args.mode == "random" else args.trials,
              tokens_per_training=8 * 128 * args.max_steps,
              total_training_tokens=(args.trials+1)*8*128*args.max_steps,
              base_config=base, available_candidates=len(pool),
              output=str(ROOT / "searches" / args.name)), indent=2))
        print("No API calls, training, or file writes.")
        return
    suite = ROOT / "searches" / args.name
    if args.resume:
        if read_json(suite / "plan.json") != plan:
            raise ValueError("Plan, source or data changed. Choose another --name without --resume.")
    else:
        # Validate authentication before creating any output.
        if args.mode != "random" and not os.environ.get("OPENAI_API_KEY"):
            raise ValueError("Set OPENAI_API_KEY locally in PowerShell. Never put it in source or logs.")
        suite.mkdir(parents=True, exist_ok=False)
        write_json(suite / "plan.json", plan)
    started = time.perf_counter()
    rows = []
    order = list(pool)
    random.Random(args.search_seed).shuffle(order)
    for trial in range(args.trials + 1):
        folder = suite / f"trial_{trial:03d}"
        folder.mkdir(exist_ok=True)
        usage, actual_model, api_seconds = None, None, 0.0
        if trial == 0:
            proposal = dict(candidate_id="baseline", hypothesis="Measure the common baseline.",
                            expected_effect="Reference for every search strategy.")
        elif (folder / "proposal.json").exists():
            proposal = read_json(folder / "proposal.json")
        elif args.mode == "random":
            proposal = dict(candidate_id=next(i for i in order if i in pool),
                            hypothesis="Uniform random choice without replacement; not an LLM hypothesis.",
                            expected_effect="No directional prediction.")
        else:
            context = build_context(base, rows, args.mode)
            body = make_request(args.model, args.max_output_tokens, pool, context)
            write_json(folder / "api_request.json", body)
            if (folder / "api_response.json").exists():
                response = read_json(folder / "api_response.json")
            else:
                key = os.environ.get("OPENAI_API_KEY")
                if not key:
                    raise ValueError("Set OPENAI_API_KEY to continue proposing experiments.")
                tick = time.perf_counter()
                response = call_openai(body, key)
                api_seconds = time.perf_counter() - tick
                write_json(folder / "api_response.json", response)
                write_json(folder / "api_timing.json", {"seconds": api_seconds})
            proposal = parse_response(response, pool)
        if trial:
            # Also validate cached proposals. Never silently replace a failed LLM choice with random search.
            parsed = parse_response(dict(status="completed", output=[dict(type="message", content=[
                     dict(type="output_text", text=json.dumps(proposal))])]), pool)
            config = dict(base, **pool[parsed["candidate_id"]])
        else:
            config = dict(base)
        write_json(folder / "proposal.json", proposal)
        if (folder / "api_response.json").exists():
            response = read_json(folder / "api_response.json")
            usage, actual_model = response.get("usage"), response.get("model")
        if (folder / "api_timing.json").exists():
            api_seconds = read_json(folder / "api_timing.json")["seconds"]
        print(f"\n[{trial}/{args.trials}] {proposal['candidate_id']}: {proposal['hypothesis']}", flush=True)
        if not (folder / "training").exists():
            train_once(config, folder, data)
        result = validated_result(folder / "training", {"config": config}, plan["source_sha256"]["train.py"])
        rows.append(dict(trial=trial, proposal=proposal, config=config,
                    final_dev_loss=result["last_measured_dev_loss"], best_dev_loss=result["best_dev_loss"],
                    training_seconds=result["training_seconds"], run_elapsed_seconds=result["run_elapsed_seconds"],
                    tokens_seen=result["tokens_seen"], api_usage=usage, actual_model=actual_model,
                    api_seconds=api_seconds, training_versions=result["versions"]))
        if trial:
            del pool[proposal["candidate_id"]]
        summary = save_progress(suite, rows, plan)
    write_json(suite / "last_invocation.json", {"completed_utc": datetime.now(timezone.utc).isoformat(),
               "elapsed_seconds": time.perf_counter() - started})
    print(f"\nBaseline: {summary['baseline_dev_loss']:.5f}; best: {summary['best_final_dev_loss']:.5f}")
    print(f"Saved summary.json, journal.csv, history.json and best_config.json: {suite}")


def compare(names):
    loaded = []
    for name in names:
        folder = ROOT / "searches" / name
        plan, summary = read_json(folder / "plan.json"), read_json(folder / "summary.json")
        signature = {key: plan[key] for key in
                     ("base_config", "search_space", "trials", "data_sha256", "source_sha256")}
        if summary["status"] != "completed":
            raise ValueError(f"Search {name} is incomplete.")
        if loaded and signature != loaded[0][2]:
            raise ValueError("Searches have different budgets, training seeds, code, or data; cannot compare directly.")
        if loaded and abs(summary["baseline_dev_loss"]-loaded[0][1]["baseline_dev_loss"]) > 1e-4:
            raise ValueError("Common baseline does not reproduce; investigate before comparing.")
        loaded.append((name, summary, signature))
    print("Name                     Mode         Baseline    Best dev    Reduction   API output tokens")
    for name, s, _ in loaded:
        print(f"{name:<24} {s['mode']:<12} {s['baseline_dev_loss']:<11.5f} "
              f"{s['best_final_dev_loss']:<11.5f} {s['loss_reduction']:<11.5f} {s['total_api_output_tokens']}")
    print("Pilot on one training seed. Repeat independent searches and validate winning recipes on new seeds.")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["agent", "no-history", "random"], default="agent")
    p.add_argument("--name", default="agent_01")
    p.add_argument("--model", default="gpt-6-luna")
    p.add_argument("--trials", type=int, default=5, help="New recipes, plus one common baseline.")
    p.add_argument("--threads", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=512)
    p.add_argument("--seed", type=int, default=42, help="Training seed; keep matched across methods.")
    p.add_argument("--search-seed", type=int, default=123, help="Random search order, not an API determinism guarantee.")
    p.add_argument("--max-output-tokens", type=int, default=2048)
    p.add_argument("--data-dir", type=Path, default=ROOT / "data" / "processed")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--compare", nargs="+")
    args = p.parse_args()
    for name in [args.name, *(args.compare or [])]:
        if name in ("", ".", "..") or any(c in name for c in '/\\:'):
            p.error("Search names must be folder names.")
    if not 1 <= args.trials <= 30 or min(args.threads, args.max_steps) < 1 or min(args.seed, args.search_seed) < 0:
        p.error("Use 1..30 trials, positive steps/threads, nonnegative seeds.")
    if not 256 <= args.max_output_tokens <= 8192:
        p.error("--max-output-tokens must be 256..8192.")
    if args.compare:
        compare(args.compare)
    else:
        search(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped. Completed runs/proposals are preserved. Use --resume; move incomplete training folders aside first.", file=sys.stderr)
        sys.exit(130)
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, StopIteration) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
