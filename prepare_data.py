#!/usr/bin/env python3
"""Prepare the published BabyLM Strict-Small corpus for local experiments.

Run from the project folder:
    python prepare_data.py --download

Dependencies: numpy, tokenizers>=0.21,<1, huggingface_hub (for --download).
This script prepares data. It does not train a language model.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import shutil
import sys
import tempfile
import unicodedata


REPO_ID = "BabyLM-community/BabyLM-2026-Strict-Small"
REVISION = "c92ab16b4f08858304b0815706065b3354d8fc0a"
RAW_FILENAMES = (
    "bnc_spoken.train.txt",
    "childes.train.txt",
    "gutenberg.train.txt",
    "open_subtitles.train.txt",
    "simple_wiki.train.txt",
    "switchboard.train.txt",
)
SPLITS = ("train", "dev", "test")
SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]
PROJECT_ROOT = Path(__file__).resolve().parent


def file_sha256(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def split_for(text: str, seed: int, dev_fraction: float, test_fraction: float) -> str:
    """Keep identical NFC/whitespace-normalized lines in the same split."""
    key = " ".join(unicodedata.normalize("NFC", text).split())
    digest = hashlib.sha256(str(seed).encode("ascii") + b"\0" + key.encode("utf-8")).digest()
    position = int.from_bytes(digest[:8], "big") / (1 << 64)
    if position < dev_fraction:
        return "dev"
    if position < dev_fraction + test_fraction:
        return "test"
    return "train"


def download_raw(raw_dir: Path) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise RuntimeError("Install huggingface_hub to use --download.") from error
    print(f"Downloading/checking the pinned dataset revision: {REVISION}", flush=True)
    try:
        snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            revision=REVISION,
            local_dir=str(raw_dir),
            allow_patterns=["*.txt", "README.md"],
            max_workers=2,
        )
    except Exception as error:
        raise RuntimeError(
            f"Dataset download failed: {error}. "
            "If all six raw files are already present, run without --download."
        ) from error


def prepare_text_splits(raw_dir: Path, work_dir: Path, args: argparse.Namespace) -> tuple[dict, dict]:
    totals = {name: {"lines": 0} for name in SPLITS}
    sources = {}
    with ExitStack() as stack:
        outputs = {
            name: stack.enter_context((work_dir / f"{name}.txt").open("w", encoding="utf-8", newline="\n"))
            for name in SPLITS
        }
        for filename in RAW_FILENAMES:
            path = raw_dir / filename
            stats = {"sha256": file_sha256(path), "bytes": path.stat().st_size,
                     "raw_lines": 0, "empty_lines_skipped": 0,
                     "split_lines": {name: 0 for name in SPLITS}}
            with path.open("r", encoding="utf-8-sig", errors="strict") as handle:
                for line in handle:
                    stats["raw_lines"] += 1
                    text = line.strip()
                    if not text:
                        stats["empty_lines_skipped"] += 1
                        continue
                    name = split_for(text, args.seed, args.dev_fraction, args.test_fraction)
                    outputs[name].write(text + "\n")
                    totals[name]["lines"] += 1
                    stats["split_lines"][name] += 1
            sources[filename] = stats
            print(f"  Read {filename}: {stats['raw_lines']:,} lines", flush=True)
    for name in SPLITS:
        if totals[name]["lines"] == 0:
            raise RuntimeError(f"The {name} split is empty. Check the input files and split fractions.")
    return totals, sources


def train_tokenizer(train_path: Path, tokenizer_path: Path, vocab_size: int):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    # Never train the vocabulary on dev/test data.
    tokenizer.train(files=[str(train_path)], trainer=trainer)
    tokenizer.save(str(tokenizer_path))
    return Tokenizer.from_file(str(tokenizer_path))


def encode_split(tokenizer, text_path: Path, binary_path: Path) -> dict:
    import numpy as np
    eos_id = tokenizer.token_to_id("<eos>")
    if eos_id is None:
        raise RuntimeError("The tokenizer does not contain <eos>.")
    stats = {"tokens": 0, "lines": 0, "max_token_id": -1}

    def write_batch(batch: list[str], output) -> None:
        ids = []
        for encoding in tokenizer.encode_batch(batch, add_special_tokens=False):
            ids.extend(encoding.ids)
            ids.append(eos_id)
        if ids:
            highest = max(ids)
            if highest >= tokenizer.get_vocab_size() or highest > 65535:
                raise RuntimeError("A token ID is outside the supported vocabulary range.")
            stats["max_token_id"] = max(stats["max_token_id"], highest)
            np.asarray(ids, dtype="<u2").tofile(output)
            stats["tokens"] += len(ids)
        stats["lines"] += len(batch)

    with text_path.open("r", encoding="utf-8") as source, binary_path.open("wb") as output:
        batch = []
        for line in source:
            batch.append(line.rstrip("\n"))
            if len(batch) >= 512:
                write_batch(batch, output)
                batch = []
        if batch:
            write_batch(batch, output)
    if binary_path.stat().st_size != 2 * stats["tokens"]:
        raise RuntimeError(f"Unexpected binary file size: {binary_path.name}")
    stats["bytes"] = binary_path.stat().st_size
    stats["sha256"] = file_sha256(binary_path)
    return stats


def make_report(manifest: dict) -> str:
    lines = [
        "BabyLM data preparation completed successfully.",
        f"Source: {REPO_ID}",
        f"Reference revision: {REVISION}",
        f"Tokenizer: byte-level BPE; vocabulary={manifest['tokenizer']['vocab_size']}",
        "Binary format: little-endian uint16 (<u2), one <eos> after each non-empty input line.",
        "",
    ]
    for name in SPLITS:
        info = manifest["splits"][name]
        lines.append(f"{name}: {info['lines']:,} lines; {info['tokens']:,} BPE tokens; {info['bytes']:,} bytes")
    lines += [
        "",
        "Split assignment is fixed by the normalized line content and seed.",
        "Exact normalized duplicate lines always remain in one split.",
        "Document IDs are unavailable: this is a line-group split, not a document-disjoint split.",
        "Near-duplicate passages or neighboring lines may still appear in different splits.",
        "dev and test are our INTERNAL holdouts, not the official BabyLM evaluation tasks.",
        "Tokenizer fitting uses train.txt only. Keep all prepared files fixed across experiments.",
        "BPE token counts are not word counts and need not equal the dataset card's count.",
        "",
        "Example (an illustrative sentence, not a quote from the corpus):",
        json.dumps(manifest["example"], ensure_ascii=True),
        "",
        "Next step: train a small causal language model using train.bin, evaluate on dev.bin.",
        "Reserve test.bin and the official BabyLM suite for the final comparison.",
    ]
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", type=Path, default=PROJECT_ROOT / "data" / "raw")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "data" / "processed")
    parser.add_argument("--download", action="store_true", help="Download/check the pinned official dataset revision.")
    parser.add_argument("--vocab-size", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dev-fraction", type=float, default=0.02)
    parser.add_argument("--test-fraction", type=float, default=0.02)
    args = parser.parse_args()
    if not (512 <= args.vocab_size <= 65535):
        parser.error("--vocab-size must be between 512 and 65535.")
    if not (0 < args.dev_fraction < 1 and 0 < args.test_fraction < 1
            and args.dev_fraction + args.test_fraction < 1):
        parser.error("dev/test fractions must be positive and their sum must be less than 1.")
    return args


def main() -> None:
    args = parse_args()
    try:
        import numpy  # noqa: F401
        import tokenizers  # noqa: F401
    except ImportError as error:
        raise RuntimeError('Install dependencies first: python -m pip install numpy "tokenizers>=0.21,<1" huggingface_hub') from error

    raw_dir = args.raw_dir.resolve()
    out_dir = args.out_dir.resolve()
    if out_dir == raw_dir or out_dir.is_relative_to(raw_dir):
        raise RuntimeError("Choose --out-dir outside the raw data folder.")
    if out_dir.exists() and (not out_dir.is_dir() or any(out_dir.iterdir())):
        raise RuntimeError(f"Output is not empty: {out_dir}. Reuse the existing preparation or choose a new --out-dir.")
    if args.download:
        download_raw(raw_dir)
    missing = [name for name in RAW_FILENAMES if not (raw_dir / name).is_file()]
    if missing:
        raise RuntimeError("Missing raw data files: " + ", ".join(missing) + ". Run again with --download.")
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix=out_dir.name + "-tmp-", dir=str(out_dir.parent)))
    try:
        print("Step 1/3: creating fixed train/dev/test text splits...", flush=True)
        totals, sources = prepare_text_splits(raw_dir, work_dir, args)
        print("Step 2/3: learning a tokenizer from train.txt only...", flush=True)
        tokenizer_path = work_dir / "tokenizer.json"
        tokenizer = train_tokenizer(work_dir / "train.txt", tokenizer_path, args.vocab_size)
        print("Step 3/3: converting all three splits to token IDs...", flush=True)
        splits = {}
        for name in SPLITS:
            splits[name] = encode_split(tokenizer, work_dir / f"{name}.txt", work_dir / f"{name}.bin")
            if splits[name]["lines"] != totals[name]["lines"]:
                raise RuntimeError(f"Line count mismatch for {name}.")
            print(f"  {name}: {splits[name]['tokens']:,} BPE tokens", flush=True)
        demo_text = "The cat is sleeping."
        demo = tokenizer.encode(demo_text, add_special_tokens=False)
        if tokenizer.decode(demo.ids, skip_special_tokens=True) != demo_text:
            raise RuntimeError("The byte-level tokenizer failed the example round-trip check.")
        manifest = {
            "format_version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source": {"repo_id": REPO_ID, "reference_revision": REVISION,
                       "downloaded_pinned_revision_this_run": args.download,
                       "local_raw_files": sources},
            "preparation": {"seed": args.seed, "dev_fraction": args.dev_fraction,
                            "test_fraction": args.test_fraction,
                            "split_method": "sha256-of-seed-and-NFC-whitespace-normalized-line-v1",
                            "scope": "line groups; not document-disjoint",
                            "empty_lines": "skipped", "duplicate_lines": "kept, but assigned to the same split",
                            "internal_test_is_official_babylm_test": False},
            "tokenizer": {"type": "byte-level BPE", "trained_on": "train.txt only",
                          "requested_vocab_size": args.vocab_size,
                          "vocab_size": tokenizer.get_vocab_size(), "sha256": file_sha256(tokenizer_path),
                          "special_token_ids": {text: tokenizer.token_to_id(text) for text in SPECIAL_TOKENS}},
            "binary": {"dtype": "<u2", "eos_after_each_line": True, "suggested_initial_context_length": 128},
            "splits": splits,
            "versions": {"python": platform.python_version(), "numpy": version("numpy"),
                         "tokenizers": version("tokenizers")},
            "script_sha256": file_sha256(Path(__file__)),
            "example": {"text": demo_text, "tokens": demo.tokens, "ids": demo.ids},
        }
        if args.download:
            manifest["versions"]["huggingface_hub"] = version("huggingface_hub")
        (work_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        report = make_report(manifest)
        (work_dir / "data_report.txt").write_text(report, encoding="utf-8")
        if out_dir.exists():
            out_dir.rmdir()  # Only an empty output directory is permitted above.
        work_dir.replace(out_dir)
        print("\n" + report, flush=True)
        print(f"Saved to: {out_dir}", flush=True)
    finally:
        if work_dir.exists():
            shutil.rmtree(work_dir)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
