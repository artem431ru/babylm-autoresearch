#!/usr/bin/env python3
"""Train a small causal Transformer on data produced by prepare_data.py.

First CPU pilot:
    python train.py --run-name baseline_001 --max-steps 100

All model weights start from scratch. Only train.bin and dev.bin are read.
The fixed sampled dev metric is a local proxy, not an official BabyLM score.
Dependencies: torch, numpy; matplotlib for the optional learning-curve image.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import sys
import time

try:
    import numpy as np
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError as error:
    raise SystemExit("Missing dependency: install PyTorch and NumPy in this Python environment.") from error


PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass
class TrainConfig:
    max_steps: int = 100
    batch_size: int = 4
    block_size: int = 128
    n_layer: int = 2
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_fraction: float = 0.1
    min_lr_ratio: float = 0.1
    grad_clip: float = 1.0
    eval_interval: int = 25
    eval_batches: int = 20
    eval_batch_size: int = 4
    eval_seed: int = 12345
    log_interval: int = 10
    seed: int = 42
    threads: int = 4
    device: str = "cpu"


class CausalAttention(nn.Module):
    def __init__(self, width: int, heads: int, dropout: float):
        super().__init__()
        self.heads = heads
        self.dropout = dropout
        self.qkv = nn.Linear(width, 3 * width)
        self.projection = nn.Linear(width, width)
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [t.reshape(batch, length, self.heads, width // self.heads).transpose(1, 2)
                   for t in (q, k, v)]
        x = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        x = x.transpose(1, 2).contiguous().reshape(batch, length, width)
        return self.output_dropout(self.projection(x))


class Block(nn.Module):
    def __init__(self, width: int, heads: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(width)
        self.attention = CausalAttention(width, heads, dropout)
        self.ln2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, 4 * width), nn.GELU(approximate="tanh"),
            nn.Linear(4 * width, width), nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attention(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class TinyGPT(nn.Module):
    def __init__(self, vocab_size: int, block_size: int = 128, n_layer: int = 2,
                 n_head: int = 4, n_embd: int = 128, dropout: float = 0.1):
        super().__init__()
        if n_embd % n_head:
            raise ValueError("n_embd must be divisible by n_head.")
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.input_dropout = nn.Dropout(dropout)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, dropout) for _ in range(n_layer)])
        self.final_norm = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.apply(self._init_weights)
        # The input embedding and output prediction share the same matrix.
        self.lm_head.weight = self.token_embedding.weight

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, tokens, targets=None):
        if tokens.shape[1] > self.block_size:
            raise ValueError("Input sequence is longer than block_size.")
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        x = self.input_dropout(self.token_embedding(tokens) + self.position_embedding(positions))
        logits = self.lm_head(self.final_norm(self.blocks(x)))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        return logits, loss


def sha256_file(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def write_json(path: Path, obj: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def save_checkpoint(path: Path, model, optimizer, model_config: dict, config: TrainConfig,
                    step: int, tokens_seen: int, dev_loss: float, data_identity: dict, batch_rng) -> None:
    temporary = path.with_name(path.name + ".tmp")
    torch.save({
        "format_version": 1, "model_config": model_config, "training_config": asdict(config),
        "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
        "step": step, "tokens_seen": tokens_seen, "dev_loss": dev_loss,
        "data_identity": data_identity, "torch_version": str(torch.__version__),
        "cpu_rng_state": torch.get_rng_state(), "batch_rng_state": batch_rng.get_state(),
    }, temporary)
    temporary.replace(path)


def load_data(data_dir: Path, block_size: int):
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Missing {manifest_path}. Run prepare_data.py first.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("binary", {}).get("dtype") != "<u2":
        raise ValueError("Expected uint16 (<u2) files from prepare_data.py.")
    identity = {"manifest_sha256": sha256_file(manifest_path), "splits": {}}
    streams = {}
    vocab_size = int(manifest["tokenizer"]["vocab_size"])
    for name in ("train", "dev"):
        path = data_dir / f"{name}.bin"
        expected = manifest["splits"][name]
        if not path.is_file() or path.stat().st_size != 2 * int(expected["tokens"]):
            raise ValueError(f"Missing or incorrect-sized file: {path}")
        checksum = sha256_file(path)
        if checksum != expected["sha256"]:
            raise ValueError(f"Data checksum mismatch: {path}")
        stream = np.memmap(path, dtype="<u2", mode="r")
        if len(stream) <= block_size:
            raise ValueError(f"Not enough {name} tokens for block_size={block_size}.")
        if int(stream.max()) >= vocab_size:
            raise ValueError(f"Invalid token ID in {path}")
        streams[name] = stream
        identity["splits"][name] = {"sha256": checksum, "tokens": len(stream)}
    tokenizer_path = data_dir / "tokenizer.json"
    tokenizer_sha = sha256_file(tokenizer_path)
    if tokenizer_sha != manifest["tokenizer"]["sha256"]:
        raise ValueError("The saved tokenizer has changed since data preparation.")
    identity["tokenizer_sha256"] = tokenizer_sha
    identity["dataset_reference"] = manifest["source"]
    return streams, vocab_size, identity


def batch_at(stream, starts, block_size: int, device):
    sequences = np.stack([np.array(stream[int(s):int(s) + block_size + 1], dtype=np.int64) for s in starts])
    tokens = torch.from_numpy(sequences)
    return tokens[:, :-1].to(device), tokens[:, 1:].to(device)


def fixed_dev_starts(length: int, config: TrainConfig):
    # This generator is independent from model initialization and training sampling.
    rng = np.random.default_rng(config.eval_seed)
    return rng.integers(0, length - config.block_size,
                        size=(config.eval_batches, config.eval_batch_size), dtype=np.int64)


@torch.inference_mode()
def evaluate_dev(model, stream, starts, block_size: int, device) -> float:
    was_training = model.training
    model.eval()
    total_loss, count = 0.0, 0
    try:
        for positions in starts:
            x, y = batch_at(stream, positions, block_size, device)
            _, loss = model(x, y)
            total_loss += float(loss.item()) * y.numel()
            count += y.numel()
    finally:
        model.train(was_training)
    value = total_loss / count
    if not math.isfinite(value):
        raise ValueError("Non-finite dev loss.")
    return value


def learning_rate_at(step: int, config: TrainConfig) -> float:
    warmup = max(1, round(config.max_steps * config.warmup_fraction)) if config.warmup_fraction else 0
    if step <= warmup:
        return config.learning_rate * step / warmup
    progress = min(1.0, (step - warmup) / max(1, config.max_steps - warmup))
    factor = config.min_lr_ratio + 0.5 * (1 - config.min_lr_ratio) * (1 + math.cos(math.pi * progress))
    return config.learning_rate * factor


def plot_curve(rows: list[dict], path: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Plot skipped: install matplotlib to create loss_curve.png.", flush=True)
        return False
    train = [r for r in rows if r.get("train_loss") is not None]
    dev = [r for r in rows if r.get("dev_loss") is not None]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot([r["step"] for r in train], [r["train_loss"] for r in train],
            color="#2367B5", alpha=0.65, linewidth=1, label="Train: one batch, before update")
    ax.plot([r["step"] for r in dev], [r["dev_loss"] for r in dev],
            color="#C44E36", marker="o", linewidth=2, label="Dev: fixed sampled windows")
    ax.set(xlabel="Optimizer step", ylabel="Next-token cross-entropy (nats)",
           title="TinyGPT pilot learning curve")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def validate_config(config: TrainConfig) -> None:
    positive_ints = ["max_steps", "batch_size", "block_size", "n_layer", "n_head", "n_embd",
                     "eval_interval", "eval_batches", "eval_batch_size", "log_interval", "threads"]
    for name in positive_ints:
        value = getattr(config, name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    for name in ("seed", "eval_seed"):
        if not isinstance(getattr(config, name), int) or getattr(config, name) < 0:
            raise ValueError(f"{name} must be a non-negative integer.")
    if config.n_embd % config.n_head:
        raise ValueError("n_embd must be divisible by n_head.")
    for name in ("dropout", "learning_rate", "weight_decay", "warmup_fraction", "min_lr_ratio", "grad_clip"):
        value = getattr(config, name)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number.")
    if not 0 <= config.dropout < 1:
        raise ValueError("dropout must be in [0, 1).")
    if config.learning_rate <= 0 or config.weight_decay < 0 or config.grad_clip <= 0:
        raise ValueError("learning_rate and grad_clip must be positive; weight_decay must be non-negative.")
    if not 0 <= config.warmup_fraction < 1 or not 0 <= config.min_lr_ratio <= 1:
        raise ValueError("Invalid warmup_fraction or min_lr_ratio.")
    if config.device not in ("cpu", "cuda", "auto"):
        raise ValueError("device must be cpu, cuda or auto.")


def run_training(config: TrainConfig, data_dir: Path, out_dir: Path) -> dict:
    validate_config(config)
    if out_dir.exists():
        raise ValueError(f"Run directory already exists: {out_dir}. Choose another --run-name or --out-dir.")
    torch.set_num_threads(config.threads)
    random.seed(config.seed)
    np.random.seed(config.seed % (2 ** 32))
    torch.manual_seed(config.seed)
    chosen_device = ("cuda" if torch.cuda.is_available() else "cpu") if config.device == "auto" else config.device
    device = torch.device(chosen_device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA/ROCm PyTorch device is unavailable. Use --device cpu.")
        torch.cuda.manual_seed_all(config.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    started = time.perf_counter()
    streams, vocab_size, data_identity = load_data(data_dir, config.block_size)
    model_config = {"vocab_size": vocab_size, "block_size": config.block_size,
                    "n_layer": config.n_layer, "n_head": config.n_head,
                    "n_embd": config.n_embd, "dropout": config.dropout}
    model = TinyGPT(**model_config).to(device)
    groups = [
        {"params": [p for p in model.parameters() if p.requires_grad and p.ndim >= 2],
         "weight_decay": config.weight_decay},
        {"params": [p for p in model.parameters() if p.requires_grad and p.ndim < 2],
         "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(groups, lr=config.learning_rate, betas=(0.9, 0.95), foreach=False)
    batch_rng = torch.Generator(device="cpu").manual_seed(config.seed + 1000)
    dev_starts = fixed_dev_starts(len(streams["dev"]), config)
    parameter_count = sum(p.numel() for p in model.parameters())
    out_dir.mkdir(parents=True)
    write_json(out_dir / "config.json", asdict(config))
    write_json(out_dir / "data_identity.json", data_identity)
    write_json(out_dir / "eval_windows.json", {"eval_seed": config.eval_seed, "starts": dev_starts.tolist(),
               "block_size": config.block_size, "batch_size": config.eval_batch_size,
               "note": "Fixed sampled windows from our internal dev split; no test data is read."})
    print(f"Device: {device}; model parameters: {parameter_count:,}; dtype: float32", flush=True)
    print(f"Train stream: {len(streams['train']):,} tokens; evaluation: {dev_starts.size * config.block_size:,} dev tokens", flush=True)
    rows = []
    tokens_seen, step_done, best_step = 0, 0, 0
    training_seconds, evaluation_seconds = 0.0, 0.0
    best_dev_loss, last_dev_loss, initial_dev_loss = math.inf, math.inf, math.inf
    status = "completed"
    columns = ["step", "tokens_seen", "train_loss", "dev_loss", "learning_rate", "grad_norm",
               "step_seconds", "training_seconds", "eval_seconds", "train_tokens_per_second"]
    try:
        with (out_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            eval_start = time.perf_counter()
            initial_dev_loss = last_dev_loss = best_dev_loss = evaluate_dev(
                model, streams["dev"], dev_starts, config.block_size, device)
            evaluation_seconds += time.perf_counter() - eval_start
            initial_row = {"step": 0, "tokens_seen": 0, "train_loss": None, "dev_loss": initial_dev_loss,
                           "learning_rate": 0.0, "grad_norm": None, "step_seconds": 0.0,
                           "training_seconds": 0.0, "eval_seconds": evaluation_seconds,
                           "train_tokens_per_second": 0.0}
            writer.writerow(initial_row)
            handle.flush()
            rows.append(initial_row)
            save_checkpoint(out_dir / "best.pt", model, optimizer, model_config, config,
                            0, 0, initial_dev_loss, data_identity, batch_rng)
            print(f"Before training | dev_loss={initial_dev_loss:.4f}", flush=True)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            model.train()
            for step in range(1, config.max_steps + 1):
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                tick = time.perf_counter()
                positions = torch.randint(0, len(streams["train"]) - config.block_size,
                                          (config.batch_size,), generator=batch_rng).tolist()
                x, y = batch_at(streams["train"], positions, config.block_size, device)
                lr = learning_rate_at(step, config)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                optimizer.zero_grad(set_to_none=True)
                _, loss = model(x, y)
                train_loss = float(loss.item())
                if not math.isfinite(train_loss):
                    raise ValueError("Training loss became non-finite.")
                loss.backward()
                grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip).item())
                if not math.isfinite(grad_norm):
                    raise ValueError("Gradient norm became non-finite.")
                optimizer.step()
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                step_seconds = time.perf_counter() - tick
                training_seconds += step_seconds
                tokens_seen += y.numel()
                step_done = step
                dev_loss, eval_seconds = None, 0.0
                if step % config.eval_interval == 0 or step == config.max_steps:
                    eval_start = time.perf_counter()
                    dev_loss = last_dev_loss = evaluate_dev(model, streams["dev"], dev_starts, config.block_size, device)
                    eval_seconds = time.perf_counter() - eval_start
                    evaluation_seconds += eval_seconds
                    if dev_loss < best_dev_loss:
                        best_dev_loss, best_step = dev_loss, step
                        save_checkpoint(out_dir / "best.pt", model, optimizer, model_config, config,
                                        step, tokens_seen, dev_loss, data_identity, batch_rng)
                row = {"step": step, "tokens_seen": tokens_seen, "train_loss": train_loss, "dev_loss": dev_loss,
                       "learning_rate": lr, "grad_norm": grad_norm, "step_seconds": step_seconds,
                       "training_seconds": training_seconds, "eval_seconds": eval_seconds,
                       "train_tokens_per_second": tokens_seen / training_seconds}
                writer.writerow(row)
                rows.append(row)
                if step % config.log_interval == 0 or dev_loss is not None:
                    handle.flush()
                    extra = f" | dev_loss={dev_loss:.4f}" if dev_loss is not None else ""
                    print(f"step {step}/{config.max_steps} | train_loss={train_loss:.4f}{extra} | "
                          f"{tokens_seen / training_seconds:,.0f} train tok/s", flush=True)
    except KeyboardInterrupt:
        status = "interrupted"
        print("Interrupted. Previously measured best.pt and completed metric rows remain available.", flush=True)
    except Exception as error:
        write_json(out_dir / "failure.json", {"status": "failed", "completed_steps": step_done, "error": str(error)})
        raise
    if not math.isfinite(initial_dev_loss):
        result = {"status": status, "completed_steps": step_done,
                  "note": "Interrupted before the initial dev evaluation completed; no model comparison is available.",
                  "config": asdict(config)}
        write_json(out_dir / "results.json", result)
        return result
    if status == "completed":
        save_checkpoint(out_dir / "last.pt", model, optimizer, model_config, config,
                        step_done, tokens_seen, last_dev_loss, data_identity, batch_rng)
    plot_error = None
    try:
        figure_saved = plot_curve(rows, out_dir / "loss_curve.png")
    except Exception as error:
        figure_saved, plot_error = False, str(error)
        print(f"Plot could not be saved: {error}. Metrics and checkpoints remain available.", flush=True)
    result = {
        "status": status, "created_utc": datetime.now(timezone.utc).isoformat(),
        "completed_steps": step_done, "tokens_seen": tokens_seen,
        "train_stream_tokens": len(streams["train"]),
        "token_budget_over_train_stream_size": tokens_seen / len(streams["train"]),
        "initial_dev_loss": initial_dev_loss, "last_measured_dev_loss": last_dev_loss,
        "best_dev_loss": best_dev_loss, "best_step": best_step,
        "dev_metric": "mean next-token cross-entropy, nats, on fixed sampled internal dev windows",
        "dev_eval_tokens": int(dev_starts.size * config.block_size),
        "training_seconds": training_seconds, "evaluation_seconds": evaluation_seconds,
        "train_tokens_per_second": tokens_seen / training_seconds if training_seconds else 0.0,
        "run_elapsed_seconds": time.perf_counter() - started,
        "peak_gpu_memory_mb": torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device.type == "cuda" else None,
        "device": str(device), "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "parameter_count": parameter_count, "dtype": "float32", "figure_saved": figure_saved,
        "plot_error": plot_error,
        "versions": {"python": platform.python_version(), "torch": str(torch.__version__), "numpy": str(np.__version__)},
        "script_sha256": sha256_file(Path(__file__)),
        "seed_note": "Seeds are fixed. Identical results across different hardware/software stacks are not guaranteed.",
        "config": asdict(config),
    }
    write_json(out_dir / "results.json", result)
    print(f"\n{status.upper()} | initial dev_loss={initial_dev_loss:.4f} | "
          f"best dev_loss={best_dev_loss:.4f} at step {best_step}", flush=True)
    print(f"Training speed: {result['train_tokens_per_second']:,.0f} tok/s; output: {out_dir}", flush=True)
    return result


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, help="Optional JSON recipe. Explicit command-line options override it.")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "processed")
    parser.add_argument("--run-name", default="baseline_001")
    parser.add_argument("--out-dir", type=Path, help="Alternative output directory; must not already exist.")
    defaults = TrainConfig()
    for item in fields(TrainConfig):
        default = getattr(defaults, item.name)
        parser.add_argument("--" + item.name.replace("_", "-"), dest=item.name, type=type(default), default=None)
    args = parser.parse_args()
    recipe = asdict(defaults)
    if args.config:
        incoming = json.loads(args.config.read_text(encoding="utf-8"))
        if not isinstance(incoming, dict) or set(incoming) - set(recipe):
            parser.error("Config must be a JSON object containing only known training parameters.")
        recipe.update(incoming)
    for name in recipe:
        value = getattr(args, name)
        if value is not None:
            recipe[name] = value
    if args.run_name in ("", ".", "..") or "/" in args.run_name or "\\" in args.run_name:
        parser.error("--run-name must be a folder name, not a path.")
    config = TrainConfig(**recipe)
    validate_config(config)
    out_dir = args.out_dir or PROJECT_ROOT / "runs" / args.run_name
    return config, args.data_dir.resolve(), out_dir.resolve()


if __name__ == "__main__":
    try:
        run_training(*parse_arguments())
    except (RuntimeError, OSError, ValueError, KeyError, TypeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
