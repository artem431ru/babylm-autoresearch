# BabyLM AutoResearch: Does Experiment History Help an LLM Agent?

Research pilot on **history-aware hyperparameter search** for a small causal language model.

The agent selects training recipes from a fixed discrete search space. We compare three methods under the same training-token budget: random search, an LLM agent without prior trial outcomes, and an LLM agent with full experiment history.

## Results

| Search method | Held-out test loss ↓ | Perplexity ↓ |
|---|---:|---:|
| Baseline | 4.5466 | 94.31 |
| Random search | 4.4547 | 86.03 |
| LLM without history | 4.4980 | 89.84 |
| **LLM with history** | **4.4022** | **81.63** |

Results are means across **three new training seeds (46, 47, 48)** after retraining the selected recipes for 2,048 steps. The history-aware agent improves test cross-entropy by **1.18% over random search** and **3.18% over baseline**. Lower is better.

![Held-out test loss by seed](figures/test_loss_by_seed.png)

**Important:** This is an **internal held-out line-group split**, not a document-disjoint test or an official BabyLM benchmark score. We do not claim generalization beyond this setup.

## Method

- Architecture: 2-layer causal Transformer, 4 attention heads, embedding dimension 128, approximately 1.44M parameters.
- Search space: learning rate, weight decay, dropout and warmup fraction (108 total combinations including baseline).
- Search: 5 new recipes + common baseline per method and seed; repeated on seeds 43, 44, 45.
- Fixed short-run budget: 512 steps × batch 8 × context 128 = 524,288 training tokens per recipe.
- Recipe selection: best final internal dev loss for each method, selected **before** looking at test results.
- Transfer: 2,048 training steps from scratch, on three new seeds.
- Evaluation: fixed held-out test set, 343,546 scored tokens, next-token cross-entropy.
- LLM used for the pilot: `gpt-6-luna` through a third-party Responses-compatible gateway.

## Main finding

Giving the LLM agent the outcomes of earlier trials helped it search more effectively in this pilot. Full history was not free: it consumed approximately **28.6% more API tokens** than the no-history LLM condition. A compact-history ablation is the next planned experiment.

## Documentation and evidence

- [`REPORT_RU.md`](docs/REPORT_RU.md) — complete Russian research report.
- [`REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) — commands, hashes, assumptions and missing dependencies.
- [`NEXT_EXPERIMENT.md`](docs/NEXT_EXPERIMENT.md) — preregistered plan for compact-history ablation.
- `results/crs_repeat_comparison.json` — short search results.
- `results/long_validation.json` — long validation results.
- `results/test_results.json` — held-out test results.
- `results/tables/` — derived CSVs.

To reproduce the reported tables **offline**, without training or an API key:

```bash
python scripts/analyze_results.py
```

For full training replication, use the repository source files (`train.py`, `runner.py`, `agent_crs.py`, `prepare_data.py`, `validate_long.py`, `evaluate_test.py`) and recreate the exact processed dataset. The original training/evaluation checkpoints are not distributed. Code and data hashes are recorded in the results files.

## Reproduction and safety notes

Searches can incur paid LLM API usage; a `--dry-run` mode exists. Never commit API keys, raw gateway responses containing sensitive information, checkpoints with private data, or personal paths. The gateway in the original experiment was third-party, not an official OpenAI endpoint.
