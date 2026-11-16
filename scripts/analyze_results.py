#!/usr/bin/env python3
"""Offline summary of three already completed BabyLM experiments.

Run from the repository root: python scripts/analyze_results.py
Standard library only. Does not train, access the API, or use test for selection.
"""
from __future__ import annotations
import csv
import json
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'results'
TABLES = ROOT / 'results' / 'tables'


def load(name):
    return json.loads((EVIDENCE / name).read_text(encoding='utf-8'))


def save(name, rows, columns):
    TABLES.mkdir(parents=True, exist_ok=True)
    with (TABLES / name).open('w', encoding='utf-8-sig', newline='') as out:
        writer = csv.DictWriter(out, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main():
    search = load('crs_repeat_comparison.json')
    validation = load('long_validation.json')
    test = load('test_results.json')
    assert len(search) == 9
    assert validation['completed_trainings'] == 12
    assert test['completed_evaluations'] == 12
    assert test['plan']['checkpoint_selection'].startswith('last.pt')

    search_rows = []
    for method in ('random', 'no-history', 'agent'):
        entries = sorted((x for x in search if x['summary']['mode'] == method),
                         key=lambda x: x['summary']['training_seed'])
        assert [x['summary']['training_seed'] for x in entries] == [43, 44, 45]
        for e in entries:
            s = e['summary']
            search_rows.append(dict(method=method, seed=s['training_seed'],
                best_final_dev_loss=s['best_final_dev_loss'],
                baseline_dev_loss=s['baseline_dev_loss'],
                api_input_tokens=s['total_api_input_tokens'],
                api_output_tokens=s['total_api_output_tokens'],
                trainings=s['completed_trainings']))
    save('search_runs.csv', search_rows, list(search_rows[0]))

    val_rows = [dict(method=r['method'], seed=r['seed'],
        final_dev_loss=r['result']['last_measured_dev_loss'],
        tokens_seen=r['result']['tokens_seen']) for r in validation['runs']]
    save('long_validation.csv', sorted(val_rows, key=lambda r:(r['seed'],r['method'])),
         list(val_rows[0]))

    test_rows = [dict(method=r['method'], seed=r['seed'], test_loss=r['test_loss'],
        perplexity=r['perplexity'], scored_tokens=r['scored_tokens']) for r in test['runs']]
    save('test_by_seed.csv', sorted(test_rows, key=lambda r:(r['seed'],r['method'])),
         list(test_rows[0]))
    summary_rows = []
    for method in ('baseline','random','no-history','agent'):
        vals = [r['test_loss'] for r in test_rows if r['method'] == method]
        ppls = [r['perplexity'] for r in test_rows if r['method'] == method]
        assert len(vals) == 3
        assert abs(mean(vals)-test['summary'][method]['mean_test_loss']) < 1e-10
        summary_rows.append(dict(method=method, seeds='46;47;48', n=3,
            mean_test_loss=mean(vals), sample_std_test_loss=stdev(vals),
            mean_perplexity=mean(ppls)))
    save('test_summary.csv', summary_rows, list(summary_rows[0]))
    baseline = summary_rows[0]['mean_test_loss']
    random = summary_rows[1]['mean_test_loss']
    agent = summary_rows[3]['mean_test_loss']
    print('TEST LOSS:')
    for r in summary_rows:
        print(f"{r['method']:<11} {r['mean_test_loss']:.6f} ± {r['sample_std_test_loss']:.6f}")
    print(f'Agent relative improvement vs baseline: {(baseline-agent)/baseline*100:.3f}%')
    print(f'Agent relative improvement vs random: {(random-agent)/random*100:.3f}%')
    print('Offline summary complete; no training, API calls or new test evaluations.')


if __name__ == '__main__':
    main()
