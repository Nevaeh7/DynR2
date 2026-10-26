#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

case "${1:-}" in
  Beauty) domain=Beauty; name=beauty ;;
  Sports) domain=Sports; name=sports ;;
  Toys) domain=Toys; name=toys ;;
  *) printf 'Usage: bash scripts/run.sh {Beauty|Sports|Toys} [train|test|all]\n' >&2; exit 2 ;;
esac

stage="${2:-all}"
case "$stage" in
  train|test|all) ;;
  *) printf 'Stage must be train, test, or all.\n' >&2; exit 2 ;;
esac

python_bin="${PYTHON:-python}"
gpus="${GPUS:-4}"
case "$gpus" in
  1|2|4|8) ;;
  *) printf 'GPUS must be 1, 2, 4, or 8.\n' >&2; exit 2 ;;
esac

artifact_root="${DYNR2_ARTIFACT_ROOT:-$(dirname "$PWD")/$(basename "$PWD")-artifacts}"
export DYNR2_ARTIFACT_ROOT="$artifact_root"
model="$artifact_root/models/$domain/initial"
data="data/$domain"
train_output="$artifact_root/outputs/${name}_train"
test_output="$artifact_root/outputs/${name}_test"

if [[ ! -d "$model" ]]; then
  printf 'Missing initial SID-trained model: %s\n' "$model" >&2
  exit 1
fi
files=(item_mapping.json train_history_counts.json)
if [[ "$stage" != test ]]; then
  files+=(train.parquet model_selection.parquet)
fi
if [[ "$stage" != train ]]; then
  files+=(test.parquet)
fi
for file in "${files[@]}"; do
  if [[ ! -f "$data/$file" ]]; then
    printf 'Missing input: %s/%s\n' "$data" "$file" >&2
    exit 1
  fi
done
if [[ "$stage" != test && -e "$train_output" ]]; then
  printf 'Training output already exists: %s.\n' "$train_output" >&2
  exit 1
fi
if [[ "$stage" != train && -e "$test_output" ]]; then
  printf 'Test output already exists: %s.\n' "$test_output" >&2
  exit 1
fi
if [[ "$stage" == test && ! -f "$train_output/training.json" ]]; then
  printf 'Missing checkpoint-selection record: %s/training.json. Run training first.\n' \
    "$train_output" >&2
  exit 1
fi

if [[ "$stage" != test ]]; then
  "$python_bin" -m torch.distributed.run --standalone --nproc_per_node="$gpus" \
    --module dynr2.train \
    --model "$model" \
    --catalog "$data/item_mapping.json" \
    --config "configs/$name.json" \
    --train "$data/train.parquet" \
    --validation "$data/model_selection.parquet" \
    --history-counts "$data/train_history_counts.json" \
    --epochs 3 \
    --seed 1427 \
    --output "$train_output"
  printf 'Training summary: %s/training.json\n' "$train_output"
fi

if [[ "$stage" != train ]]; then
  selected="$("$python_bin" -c \
    'import json, sys; print(json.load(open(sys.argv[1]))["selected_adapter"])' \
    "$train_output/training.json")"

  "$python_bin" -m dynr2.evaluate \
    --model "$model" \
    --adapter "$train_output/$selected" \
    --catalog "$data/item_mapping.json" \
    --data "$data/test.parquet" \
    --history-counts "$data/train_history_counts.json" \
    --config "configs/$name.json" \
    --repeats 3 \
    --output "$test_output"

  printf 'Results: %s/metrics.json\n' "$test_output"
fi
