# Reproducibility checklist

## Scope of this repository

This repository contains the original training and agent source files together with the reported experiment summaries, an offline analysis script, CSV tables and a figure. Large processed data files and checkpoints are intentionally excluded from Git. Offline summaries are reproducible directly from the committed JSON files; rerunning training and test evaluation requires regenerating the dataset and training checkpoints.

## Verify the reported numbers without training

From the unpacked folder:

```powershell
python scripts/analyze_results.py
```

Expected mean test loss: baseline 4.546605, random 4.454673, no-history 4.498018, agent 4.402217. The script checks that 12 evaluations are present, recomputes the means, and writes the CSV tables. It makes no API requests and does not read or retrain the model.

## Reproduce short searches with the original code

The following examples require the **original repository**, containing `train.py`, `runner.py`, `agent_crs.py` and `data/processed/{manifest.json,tokenizer.json,train.bin,dev.bin}`. Use the original virtual environment and compare source/data hashes before claiming exact replication. Recorded environment for the longer training runs: Python 3.14.5, PyTorch 2.15.0+cpu, NumPy 2.5.4. The code uses CPU.

First inspect a no-cost preview:

```powershell
python agent_crs.py --mode random --name repro_random_s43 --model gpt-6-luna --seed 43 --search-seed 143 --trials 5 --max-steps 512 --threads 10 --dry-run
```

Then run the random baseline:

```powershell
python agent_crs.py --mode random --name repro_random_s43 --model gpt-6-luna --seed 43 --search-seed 143 --trials 5 --max-steps 512 --threads 10
```

For the two LLM methods, configure the `CRS_OAI_KEY` environment variable **outside the repository** and use a trusted compatible endpoint. The historical pilot used a third-party Responses-compatible gateway. Before sending any secret to a third-party service, verify that provider's trustworthiness and terms. Example commands (these can incur charges and may encounter HTTP 429):

```powershell
python agent_crs.py --mode no-history --name repro_nohistory_s43 --model gpt-6-luna --seed 43 --search-seed 143 --trials 5 --max-steps 512 --threads 10
python agent_crs.py --mode agent --name repro_agent_s43 --model gpt-6-luna --seed 43 --search-seed 143 --trials 5 --max-steps 512 --threads 10
```

Repeat for `--seed 44 --search-seed 144` and `--seed 45 --search-seed 145`, with unique names. To resume an interrupted run, use **the same arguments** plus `--resume`. Resume checks the stored plan and hashes; do not edit source/data between attempts. The LLM is not guaranteed to propose the same candidates across independent API runs, even with matching seeds.

## Exact experimental budgets

- Short search: 6 trainings × 524,288 tokens = **3,145,728 training tokens per method per search seed**.
- Across 3 methods × 3 search seeds: **28,311,552 training tokens** in short searches (counting the common baseline once per recorded method run).
- Long validation: 4 methods × 3 new seeds × 2,097,152 tokens = **25,165,824 training tokens**.
- Test evaluation: 12 saved checkpoints, each evaluated on the same **343,546 scored tokens**.

The API token budget is **separate** from training tokens. Measured across three short-search repetitions: agent with history 103,682 input + 2,238 output; agent without history 80,481 input + 1,885 output; random 0.

## Integrity information

Reference hashes recorded in the uploaded experiment plans:

| Artifact | SHA-256 |
|---|---|
| `train.py` | `8e18405f915bc6a7925a544a81ca25b715d0968aa9e38be74ee1d40f4589ce41` |
| `runner.py` | `479b7c73c0497c6116e6b8a91e8200e61cac2493a06482879e37c71be64a4308` |
| `agent_crs.py` | `ebf4bb86fa33f6d020f0a899f6ed2a39acedbe747a01367df375be5920d416e8` |
| `train.bin` | `a9a6117b01021834a9deb3d9aa59e42d616dd76991aaa70b2f75f4b9319b4fc0` |
| `dev.bin` | `5c5f3b433323fe91a392007b9267c39fa681aae742d6206212aebc5ed663032f` |
| `test.bin` | `4c8fbc51b21b27f55b1fcd2dfc180b11995af6eed97ed8fda49f5af52b3bb4bc` |

PowerShell verification example:

```powershell
Get-FileHash .\train.py -Algorithm SHA256
Get-FileHash .\agent_crs.py -Algorithm SHA256
Get-FileHash .\data\processed\train.bin -Algorithm SHA256
```

The original test-evaluation plan additionally records a checksum for each selected checkpoint. The held-out split is **line-group-based**, not document-disjoint. Do not use test results to choose new hyperparameters.

## What's needed for full independent reproduction

1. Original code for data preprocessing, training, long validation and held-out evaluation.
2. Original data or a redistributable dataset with a clearly documented preprocessing recipe.
3. Model checkpoints, or the ability to recreate them in a matched environment.
4. A fully specified environment file and platform details. Seeds do not guarantee bitwise reproducibility across different hardware and library versions.
5. An independent document-disjoint test and downstream evaluation for stronger generalization claims.
