#!/usr/bin/env python3
"""Independent stdlib completion gate. Checks formal result coverage."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(root=ROOT):
    path = root / 'results/formal-summary.json'
    require(path.exists(), 'formal evaluation summary missing: GPU experiment not completed')
    report = json.loads(path.read_text())
    require(report.get('formal') is True, 'summary is not formal')
    entries = report['evaluations']
    require(entries, 'no evaluations')
    expected = {'base-direct': 1, 'base-tools': 1, 'e2-lora': 1, 'cold-sft': 3, 'grpo': 3}
    for condition, count in expected.items():
        for dataset in ['gsm8k', 'hotpotqa', 'gsm-symbolic']:
            rows = [r for r in entries if r['condition'] == condition and r['dataset'] == dataset]
            require(len(rows) >= count and len({r['seed'] for r in rows}) == len(rows), f'{condition}/{dataset}: missing independent seed coverage')
            require(all(r['n'] >= 500 for r in rows), f'{condition}/{dataset}: insufficient final sample size')
            require(all(0 <= r['mean'] <= 1 and len(r['ci95']) == 2 for r in rows), 'invalid metrics')
    require(report['provenance'], 'missing provenance')
    files_seen = set()
    for source in report['provenance']:
        result_file = Path(source['result_file'])
        result_file = result_file if result_file.is_absolute() else root / result_file
        require(result_file.exists(), f'raw evaluation absent: {result_file}')
        require(str(result_file.resolve()) not in files_seen, 'duplicate source included')
        files_seen.add(str(result_file.resolve()))
        require(source['split'] == 'test', 'development result')
        require(source['git_commit'] not in {'unknown', ''} and bool(source['model_revision_resolved']), 'commit/model provenance missing')
        require(source['gpu'] != 'none', 'formal result must come from GPU run')
        rows = [json.loads(line) for line in result_file.read_text().split("\n") if line.strip()]
        require(len(rows) == source['tasks'] and len({r['task_id'] for r in rows}) == len(rows), 'raw result coverage mismatch')
        by_dataset = {}
        for row in rows:
            require(row['condition'] == source['condition'] and row['seed'] == source['seed'], 'raw row condition/seed mismatch')
            require(row['split'] == 'test' and row['source'].get('revision'), 'invalid row source')
            by_dataset.setdefault(row['dataset'], []).append(row)
        for dataset, raw in by_dataset.items():
            match = [r for r in entries if r['condition'] == source['condition'] and r['seed'] == source['seed'] and r['dataset'] == dataset]
            require(len(match) == 1, 'summary missing or duplicate metric')
            require(match[0]['n'] == len(raw), 'summary count mismatch')
            mean = sum(r['reward']['correctness'] for r in raw) / len(raw)
            require(abs(mean - match[0]['mean']) < 1e-10, 'summary mean does not match raw results')
    # Require actual training/checkpoint records for each reported GRPO seed.
    for seed in {r['seed'] for r in entries if r['condition'] == 'grpo'}:
        run_dir = root / 'results' / f'grpo-{seed}'
        run = json.loads((run_dir / 'run.json').read_text())
        require(run['config']['steps'] >= 300, 'formal training incomplete')
        logs = [json.loads(line) for line in (run_dir / 'train.jsonl').read_text().split("\n") if line.strip()]
        require(max(r['step'] for r in logs) >= run['config']['steps'], 'final GRPO step not recorded')
        require((run_dir / f"checkpoint-{run['config']['steps']:04d}" / 'adapter_config.json').exists(), 'final adapter missing')
    for filename in ['README.md', 'docs/baselines.md', 'docs/trajectory-analysis.md']:
        require((root / filename).exists() and (root / filename).stat().st_size > 20, f'missing {filename}')
    analysis = (root / 'docs/trajectory-analysis.md').read_text()
    require('PENDING' not in analysis and '训练完成后' not in analysis, 'trajectory inspection/reward-hacking analysis unfinished')
    print('E3 formal artifact and raw-result verification passed')


if __name__ == '__main__':
    try:
        verify()
    except (ValueError, KeyError, FileNotFoundError) as exc:
        print(f'INCOMPLETE: {exc}', file=sys.stderr)
        sys.exit(1)
