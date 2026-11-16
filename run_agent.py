#!/usr/bin/env python3
"""Load Codex provider settings and run the unchanged BabyLM agent_crs.py.

python run_agent.py --check-config --model gpt-6-luna
python run_agent.py --repeat --model gpt-6-luna
python run_agent.py --model gpt-6-luna --check-api

No shell evaluation, key logging, or changes to files in .codex.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import urllib.parse

ROOT = Path(__file__).resolve().parent
DEFAULT_URL = "https://bridge.gptclaudegemini.xyz/"


def read_config(directory):
    path = directory / "config.toml"
    if not path.exists():
        return {}
    try:
        import tomllib
    except ImportError:
        # The seller's simple string settings also work on Python 3.10.
        # This fallback only reads the fields used below, never executes TOML.
        return read_simple_config(path)
    try:
        return tomllib.loads(path.read_text(encoding="utf-8-sig"))
    except (ValueError, UnicodeError):
        raise ValueError("Cannot parse config.toml. Check its TOML syntax; file contents are not logged.") from None


def read_simple_config(path):
    result = {}
    section = result
    allowed = {"model_provider", "model", "base_url", "env_key", "wire_api"}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("["):
            if not re.fullmatch(r"\[[A-Za-z0-9_.-]+\]", line):
                raise ValueError("This config.toml needs Python 3.11+ for full TOML support.")
            section = result
            for key in line[1:-1].split("."):
                section = section.setdefault(key, {})
            continue
        match = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)", line)
        if not match or match[1] not in allowed:
            continue
        value = match[2].strip()
        if value.startswith('"'):
            try:
                decoded, end = json.JSONDecoder().raw_decode(value)
            except ValueError:
                raise ValueError("Invalid string setting in config.toml.") from None
            if not isinstance(decoded, str) or (value[end:].strip() and not value[end:].strip().startswith("#")):
                raise ValueError("Invalid string setting in config.toml.")
        elif value.startswith("'") and "'" in value[1:]:
            end = value.index("'", 1)
            decoded = value[1:end]
            if value[end+1:].strip() and not value[end+1:].strip().startswith("#"):
                raise ValueError("Invalid string setting in config.toml.")
        else:
            raise ValueError("Provider settings in config.toml must be quoted strings.")
        section[match[1]] = decoded
    return result


def read_dotenv_key(path, env_name):
    """Read just one dotenv assignment; no expansion or shell execution."""
    if not path.exists():
        return ""
    selected = ""
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        match = re.match(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$", line)
        if not match or match[1] != env_name:
            continue
        raw = match[2].strip()
        if raw.startswith(('"', "'")):
            quote = raw[0]
            end = raw.find(quote, 1)
            if end < 0 or (raw[end+1:].strip() and not raw[end+1:].strip().startswith("#")):
                raise ValueError("Invalid quoted API key in .env; expected one complete line.")
            selected = raw[1:end]
        else:
            selected = re.split(r"\s+#", raw, maxsplit=1)[0].strip()
    return selected.strip()


def resolve_settings(args):
    directory = args.codex_dir.expanduser().resolve()
    config = read_config(directory)
    provider_name = config.get("model_provider", "crs")
    provider = config.get("model_providers", {}).get(provider_name, {})
    if not isinstance(provider, dict):
        raise ValueError("Invalid provider section in config.toml.")
    if provider.get("wire_api", "responses") != "responses":
        raise ValueError("This launcher requires a provider with wire_api = responses.")
    env_name = args.api_key_env or provider.get("env_key", "CRS_OAI_KEY")
    base_url = args.base_url or provider.get("base_url", DEFAULT_URL)
    model = args.model or config.get("model", "gpt-5.4")
    if not all(isinstance(v, str) and v.strip() for v in (env_name, base_url, model)):
        raise ValueError("Provider model, base_url and env_key must be nonempty strings.")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", env_name):
        raise ValueError("env_key must be an environment variable name, not the key itself.")
    url = urllib.parse.urlsplit(base_url)
    if (url.scheme != "https" or not url.hostname or url.username is not None
            or url.password is not None or url.query or url.fragment):
        raise ValueError("base_url must be HTTPS without credentials, query or fragment.")
    child_env = os.environ.copy()
    key = child_env.get(env_name, "").strip()
    source = "environment"
    if not key:
        key = read_dotenv_key(directory / ".env", env_name)
        source = str(directory / ".env")
    if key:
        child_env[env_name] = key
    # Deliberately leave experiment reasoning at agent_crs.py's existing default low.
    # Codex CLI's model_reasoning_effort is a separate application setting.
    return dict(model=model, base_url=base_url, env_name=env_name, env=child_env,
                key_available=bool(key), key_source=source, directory=directory)


def command_args(settings):
    return ["--model", settings["model"], "--base-url", settings["base_url"],
            "--api-key-env", settings["env_name"]]


def run_once(settings, forwarded):
    engine = ROOT / "agent_crs.py"
    if not engine.is_file():
        raise ValueError("Put run_agent.py beside the existing agent_crs.py, train.py and runner.py.")
    return subprocess.run([sys.executable, "-u", str(engine), *command_args(settings), *forwarded],
                          cwd=ROOT, env=settings["env"], check=False).returncode


def repeat(settings):
    if settings["model"] != "gpt-6-luna":
        raise ValueError("The existing repeat series uses Luna. Add --model gpt-6-luna.")
    collected = []
    print("Repeat series: seeds 43,44,45; random/agent/no-history; 5 trials + baseline each.", flush=True)
    for seed in (43, 44, 45):
        for mode in ("random", "agent", "no-history"):
            name = f"crs_luna_repeat_{mode}_s{seed}"
            folder = ROOT / "searches" / name
            forwarded = ["--mode", mode, "--name", name, "--seed", str(seed),
                         "--search-seed", str(seed+100), "--trials", "5", "--max-steps", "512",
                         "--threads", "10", "--reasoning-effort", "low"]
            if folder.exists():
                forwarded.append("--resume")
            code = run_once(settings, forwarded)
            if code:
                print(f"Stopped at {name}. Saved results are retained. Fix the error and rerun the same command.",
                      file=sys.stderr)
                return code
            record = {"name": name}
            for field, filename in (("plan", "plan.json"), ("summary", "summary.json")):
                record[field] = json.loads((folder / filename).read_text(encoding="utf-8-sig"))
            if record["summary"].get("status") != "completed":
                raise ValueError(f"Incomplete search: {name}")
            import csv
            with (folder / "journal.csv").open(encoding="utf-8-sig", newline="") as handle:
                record["journal"] = list(csv.DictReader(handle))
            collected.append(record)
    target = ROOT / "crs_repeat_comparison.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(collected, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    temporary.replace(target)
    print(f"COMPLETED: {target}", flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False, allow_abbrev=False)
    parser.add_argument("--codex-dir", type=Path,
                        default=Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex"))
    parser.add_argument("--model")
    parser.add_argument("--base-url")
    parser.add_argument("--api-key-env")
    parser.add_argument("--repeat", action="store_true")
    parser.add_argument("--check-config", action="store_true")
    args, forwarded = parser.parse_known_args()
    if args.repeat and (args.check_config or forwarded):
        raise ValueError("Use --repeat with connection options only, e.g. --repeat --model gpt-6-luna.")
    settings = resolve_settings(args)
    if args.check_config:
        print(json.dumps(dict(config_directory=str(settings["directory"]), model=settings["model"],
                              base_url=settings["base_url"], key_variable=settings["env_name"],
                              key_available=settings["key_available"], key_source=settings["key_source"]),
                         ensure_ascii=False, indent=2))
        return 0 if settings["key_available"] else 1
    random_only = any(forwarded[i:i+2] == ["--mode", "random"] for i in range(len(forwarded)))
    local_only = "--dry-run" in forwarded or "--help" in forwarded or "-h" in forwarded or "--compare" in forwarded
    if not settings["key_available"] and (args.repeat or not (random_only or local_only)):
        raise ValueError(f"No {settings['env_name']} in environment or {settings['directory'] / '.env'}. "
                         "Save it there locally; do not send the key to chat.")
    return repeat(settings) if args.repeat else run_once(settings, forwarded)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Stopped. Saved searches can be continued by rerunning the same command.", file=sys.stderr)
        raise SystemExit(130)
    except (OSError, ValueError, TypeError, KeyError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
