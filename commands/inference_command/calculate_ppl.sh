#!/bin/bash
#SBATCH --partition=mbzuai
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive

# Usage:
#   sbatch calculate_ppl.sh [domain|total]
#
# Variations (set variables below as needed):
#   - Exp2 mixture search / optim runs (adjust EXPERIMENT_NAME, MODEL_BASE, BASE_TOKEN, SIZES, LR, BATCH_SIZE, TEMPLATE, CHECKPOINT_SUFFIX, RESULT_SUBDIR)
#   - Tulu / Orca evaluations (switch MODEL_BASE, TEMPLATE, EVAL_DATASETS, DOMAINS; see prior scripts for exact combos)
#   - Single-domain probing (shrink DOMAINS/SIZES arrays)

set -euo pipefail

MODE=${1:-domain}

module load cuda/12.4
source ~/miniconda3/bin/activate myenv

SCRIPT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

cd "${REPO_ROOT}"

export NCCL_DEBUG=WARN
export PYTORCH_NO_NVML=1
export NCCL_SOCKET_IFNAME=^docker0,lo
export NCCL_IB_HCA=mlx5
export NCCL_IB_GID_INDEX=3

mkdir -p printout/output_file printout/error_file

JOB_ID=${SLURM_JOB_ID:-manual}
STDOUT_LOG="printout/output_file/output_inference_${JOB_ID}.out"
STDERR_LOG="printout/error_file/error_inference_${JOB_ID}.err"

EXPERIMENT_NAME="exp1"
MODEL_BASE="Llama-3.1-8B"
BASE_TOKEN="660000"
LR="1.0e-5"
DOMAINS=("instr" "math" "code")
SIZES=("one" "half" "third" "double" "triple")
EVAL_DATASETS=("5000000_math_val" "5000000_code_val" "5000000_instr_val")
BATCH_SIZE=2
TEMPLATE="llama3"
CHECKPOINT_SUFFIX="_4"
RESULT_SUBDIR=""
EVAL_SCRIPT="cal_ppl.py"

if [[ "${MODE}" == "total" ]]; then
  MODEL_BASE="Llama-3.2-3B"
  LR="2.0e-5"
  EVAL_DATASETS=("math_val" "code_val" "instr_val")
  BATCH_SIZE=4
  CHECKPOINT_SUFFIX=""
  TEMPLATE="llama3"
  EVAL_SCRIPT="cal_ttl_ppl.py"
fi

CHECKPOINT_PATH="${REPO_ROOT}/saves/data_mixing/${EXPERIMENT_NAME}"
RESULT_ROOT="${REPO_ROOT}/results/data_mixing/${EXPERIMENT_NAME}"
SCRIPT_PATH="${REPO_ROOT}/scripts/${EVAL_SCRIPT}"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${STDOUT_LOG}"
}

for domain in "${DOMAINS[@]}"; do
  for size in "${SIZES[@]}"; do
    log "Run mode=${MODE} domain=${domain} size=${size}"

    other_domains=()
    for d in "${DOMAINS[@]}"; do
      if [[ "${d}" != "${domain}" ]]; then
        other_domains+=("${d}")
      fi
    done

    dataset_args=("${domain}_${size}")
    for other in "${other_domains[@]}"; do
      dataset_args+=("${other}_one")
    done

    dataset_suffix="${dataset_args[0]}"
    for ((i = 1; i < ${#dataset_args[@]}; i++)); do
      dataset_suffix+="_${dataset_args[i]}"
    done

    save_dir="${RESULT_ROOT}/${MODEL_BASE}/${BASE_TOKEN}${RESULT_SUBDIR}/${domain}"
    mkdir -p "${save_dir}"

    for eval_dataset in "${EVAL_DATASETS[@]}"; do
      model_path="${CHECKPOINT_PATH}/${MODEL_BASE}/${BASE_TOKEN}${CHECKPOINT_SUFFIX}/${domain}/${MODEL_BASE}_${dataset_suffix}_${LR}_seed42"

      torchrun "${SCRIPT_PATH}" \
        --model_name_or_path "${model_path}" \
        --save_name "${save_dir}/ppl_${eval_dataset}_${MODEL_BASE}_${dataset_suffix}.json" \
        --batch_size "${BATCH_SIZE}" \
        --dataset "${eval_dataset}" \
        --template "${TEMPLATE}" \
        --cutoff_len 4096 \
        >>"${STDOUT_LOG}" 2>>"${STDERR_LOG}"
    done
  done
done