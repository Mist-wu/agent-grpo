#!/usr/bin/env bash
# Prioritize all three training seeds; finish every deferred final-test item afterward.
set -euo pipefail
config=${1:-configs/a100.json}
start=${2:-core}
mkdir -p results/core
python - <<'PY'
from pathlib import Path
import json
rows = [json.loads(line) for line in Path('data/test.jsonl').read_text().split("\n") if line.strip()]
core = [row for row in rows if row['dataset'] in {'gsm8k', 'hotpotqa'}]
Path('data/test-core.jsonl').write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in core))
print(json.dumps({'stage':'priority-core','core_rows':len(core),'full_rows':len(rows),
                  'deferred_rows':len(rows)-len(core),'selection':'all GSM8K and HotpotQA final-test rows'}),flush=True)
PY
if [ "$start" = core ]; then
  python scripts/run.py evaluate --config "$config" --data data/test-core.jsonl --out results/core/eval-base-tools.jsonl --condition base-tools --batch-size "${E3_TOOL_BATCH_SIZE:-32}"
fi
for seed in 42 43 44; do
  python scripts/run.py collect --config "$config" --data data/train.jsonl --out "data/cold-${seed}.jsonl" --seed "$seed" --target 500
  python scripts/run.py sft --config "$config" --data "data/cold-${seed}.jsonl" --out "results/cold-${seed}" --seed "$seed"
  python scripts/run.py train --config "$config" --data data/train.jsonl --dev data/dev.jsonl --out "results/grpo-${seed}" --seed "$seed" --adapter "results/cold-${seed}/adapter"
  python scripts/run.py evaluate --config "$config" --data data/test-core.jsonl --out "results/core/eval-cold-${seed}.jsonl" --seed "$seed" --condition cold-sft --adapter "results/cold-${seed}/adapter" --batch-size "${E3_TOOL_BATCH_SIZE:-32}"
  python scripts/run.py evaluate --config "$config" --data data/test-core.jsonl --out "results/core/eval-grpo-${seed}.jsonl" --seed "$seed" --condition grpo --adapter "results/grpo-${seed}/checkpoint-0300" --batch-size "${E3_TOOL_BATCH_SIZE:-32}"
done

# Complete the original 6,000-row conditions, reusing every core generation.
python scripts/run.py evaluate --config "$config" --data data/test.jsonl --out results/eval-base-direct.jsonl --condition base-direct --batch-size "${E3_DIRECT_BATCH_SIZE:-32}"
expand_core() {
  local core=$1 full=$2
  if [ ! -f "$full" ]; then
    python - "$core" "$full" <<'PY'
from pathlib import Path
import sys,json,hashlib,shutil
source,output=map(Path,sys.argv[1:])
data=Path('data/test.jsonl').read_bytes()
rows=[json.loads(line) for line in data.decode().split("\n") if line.strip()]
meta=json.loads(Path(str(source)+'.metadata.json').read_text())
original_meta=dict(meta)
shutil.copyfile(source,output)
meta.update(data_sha256=hashlib.sha256(data).hexdigest(),tasks=len(rows),
            seeded_from_core={'path':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                              'metadata':original_meta,'generation_repeated':False})
Path(str(output)+'.metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
shutil.copyfile(str(source)+'.executions.jsonl',str(output)+'.executions.jsonl')
print(json.dumps({'stage':'expand-core-to-full-plan','input':str(source),'out':str(output),
                  'full_tasks':len(rows),'existing_rows':sum(bool(line.strip()) for line in source.read_text().split("\n")),'complete':False}),flush=True)
PY
  fi
}
expand_core results/core/eval-base-tools.jsonl results/eval-base-tools.jsonl
python scripts/run.py evaluate --config "$config" --data data/test.jsonl --out results/eval-base-tools.jsonl --condition base-tools --batch-size "${E3_TOOL_BATCH_SIZE:-32}"
for seed in 42 43 44; do
  expand_core "results/core/eval-cold-${seed}.jsonl" "results/eval-cold-${seed}.jsonl"
  python scripts/run.py evaluate --config "$config" --data data/test.jsonl --out "results/eval-cold-${seed}.jsonl" --seed "$seed" --condition cold-sft --adapter "results/cold-${seed}/adapter" --batch-size "${E3_TOOL_BATCH_SIZE:-32}"
  expand_core "results/core/eval-grpo-${seed}.jsonl" "results/eval-grpo-${seed}.jsonl"
  python scripts/run.py evaluate --config "$config" --data data/test.jsonl --out "results/eval-grpo-${seed}.jsonl" --seed "$seed" --condition grpo --adapter "results/grpo-${seed}/checkpoint-0300" --batch-size "${E3_TOOL_BATCH_SIZE:-32}"
done
mkdir -p results/scored-v2
for path in results/eval-*.jsonl; do
  python scripts/run.py rescore --data data/test.jsonl --input "$path" --out "results/scored-v2/$(basename "$path")"
done
echo 'Independent stages finished. E2 adapter comparison and final combined reporting still require their actual artifacts.'
