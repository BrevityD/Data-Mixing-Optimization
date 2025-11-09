#!/bin/bash
#SBATCH --partition=highprio
#SBATCH --time=500:00:00
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/common_env.sh"

function usage() {
  cat <<'EOF'
Usage: train_single.sh
EOF
}

[[ $# -eq 0 ]] || { usage; exit 1; }

function ensure_output_dirs() {
  mkdir -p printout/output_file printout/error_file
}

function slurm_head_info() {
  local head_node
  head_node=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
  local head_ip
  head_ip=$(getent hosts "$head_node" | awk '{ print $1 }')
  echo "${head_node};${head_ip}"
}

current_time=$(timestamp)

init_environment "cuda/12.4"
ensure_output_dirs

seeds=(${SEEDS:-42})
model_name=${MODEL_NAME:-"Llama-3.1-70B"}
dataset_name=${DATASET_NAME:-"tulu3-sft"}
lr=${LR:-2.0e-7}
nnodes=${NNODES:-16}

head_info=$(slurm_head_info)
head_node="${head_info%;*}"
head_node_ip="${head_info#*;}"
echo "Head node: ${head_node} (${head_node_ip})"

for seed in "${seeds[@]}"; do
  output_dir="saves/${model_name}/full/${model_name}_${dataset_name}_sft_${lr}_seed${seed}"
  echo "llama3-70b seed=${seed} output=${output_dir}"

  srun torchrun \
    --nproc_per_node=8 \
    --nnodes="${nnodes}" \
    --rdzv_id="$SLURM_JOB_ID" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${head_node_ip}:29500" \
    src/train.py \
    --model_name_or_path "checkpoints/${model_name}" \
    --stage sft \
    --do_train \
    --finetuning_type full \
    --deepspeed "examples/deepspeed/ds_z3_config.json" \
    --dataset "${dataset_name}" \
    --template llama3 \
    --cutoff_len 4096 \
    --overwrite_cache \
    --streaming True \
    --output_dir "${output_dir}" \
    --logging_steps 100 \
    --save_steps 100 \
    --eval_steps 100 \
    --plot_loss \
    --overwrite_output_dir \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate "${lr}" \
    --num_train_epochs 10 \
    --max_steps 1449 \
    --dispatch_batches False \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --bf16 \
    --ddp_timeout 180000000 \
    --per_device_eval_batch_size 4 \
    --eval_strategy steps \
    --eval_dataset val \
    --load_best_model_at_end True \
    --include_num_input_tokens_seen True \
    --save_total_limit 1 \
    --seed "${seed}" \
    >> "printout/output_file/output_${SLURM_JOB_ID}_${current_time}_${seed}.out" \
    2>> "printout/error_file/error_${SLURM_JOB_ID}_${current_time}_${seed}.err"
done