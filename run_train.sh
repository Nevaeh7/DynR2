#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

stage=all
category=Beauty
gpus=4
extra=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage) stage="${2:?Missing stage}"; shift 2 ;;
    --category) category="${2:?Missing category}"; shift 2 ;;
    --gpus) gpus="${2:?Missing GPU count}"; shift 2 ;;
    --epochs|--max-steps|--device|--codebook-size|--base-model)
      extra+=("$1" "${2:?Missing option value}"); shift 2 ;;
    -h|--help)
      printf 'Usage: bash run_train.sh --stage {expand|align|merge|rec|sid_routing|all} --category {Beauty|Sports|Toys|all} --gpus {1|2|4|8}\n'
      exit 0 ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; exit 2 ;;
  esac
done
case "$gpus" in 1|2|4|8) ;; *) printf 'Unsupported GPU count.\n' >&2; exit 2 ;; esac
case "$category" in
  all) categories=(Beauty Sports Toys) ;;
  Beauty|Sports|Toys) categories=("$category") ;;
  *) printf 'Unsupported category.\n' >&2; exit 2 ;;
esac
case "$stage" in
  all) stages=(expand align merge rec sid_routing) ;;
  expand|align|merge|rec|sid_routing) stages=("$stage") ;;
  *) printf 'Unsupported stage.\n' >&2; exit 2 ;;
esac
python_bin="${PYTHON:-python}"
artifact_root="${DYNR2_ARTIFACT_ROOT:-$(dirname "$PWD")/$(basename "$PWD")-artifacts}"
export DYNR2_ARTIFACT_ROOT="$artifact_root"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export WANDB_MODE=disabled

for domain in "${categories[@]}"; do
  for phase in "${stages[@]}"; do
    if [[ "$phase" == expand && -f "$artifact_root/model/Qwen3-1-7B-expand/config.json" ]]; then
      printf 'Using existing expanded vocabulary.\n'
      continue
    fi
    if [[ "$phase" == expand || "$phase" == merge || "$gpus" == 1 ]]; then
      "$python_bin" -m dynr2.initialize --stage "$phase" --category "$domain" ${extra[@]+"${extra[@]}"}
    else
      ds=()
      if [[ "${USE_DEEPSPEED:-1}" == 1 ]]; then
        ds=(--deepspeed configs/zero2.json)
      fi
      "$python_bin" -m torch.distributed.run --standalone --nproc_per_node="$gpus" \
        --module dynr2.initialize --stage "$phase" --category "$domain" ${ds[@]+"${ds[@]}"} ${extra[@]+"${extra[@]}"}
    fi
  done
done
