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
Usage: recipe_exp1.sh <config>

Available configs:
  50k          Total budget 50k, single prop1=0.125, 1 node.
  200k         Total budget 200k, full prop sweep, streaming.
  500k         Total budget 500k, full prop sweep, streaming.
  1m           Total budget 1M, full prop sweep, streaming.
  3m           Total budget 3M, full prop sweep, streaming

Override defaults with environment variables when submitting, e.g.:
  sbatch --nodes=8 recipe_exp1.sh 500k
EOF
}

[[ $# -ge 1 ]] || { usage; exit 1; }

CONFIG="$1"
shift || true

init_environment "cuda/12.1"

current_time=$(timestamp)
head_node=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
head_node_ip=$(getent hosts "$head_node" | awk '{ print $1 }')
echo "Head node: ${head_node} (${head_node_ip})"

model_name=${MODEL_NAME:-"Llama-3.2-3B"}
dataset1_base=${DATASET1_BASE:-"openmathinstruct2_5M"}
dataset2_base=${DATASET2_BASE:-"opencoder-sft"}
seeds=(${SEEDS:-42})
lr=${LR:-2.0e-5}

case "${CONFIG}" in
  50k)
    total_nums=(50000)
    prop1_values=(0.125)
    nnodes=${NNODES:-1}
    max_steps_divisor=32
    interval_divisor=10
    use_streaming=false
    num_epochs=3
    ;;
  200k)
    total_nums=(200000)
    prop1_values=(0.125 0.25 0.375 0.5 0.625 0.75 0.875)
    nnodes=${NNODES:-16}
    max_steps_divisor=512
    interval_divisor=10
    use_streaming=true
    num_epochs=""
    ;;
  500k)
    total_nums=(500000)
    prop1_values=(0.125 0.25 0.375 0.5 0.625 0.75 0.875)
    nnodes=${NNODES:-16}
    max_steps_divisor=512
    interval_divisor=10
    use_streaming=true
    num_epochs=""
    ;;
  1m)
    total_nums=(1000000)
    prop1_values=(0.125 0.25 0.375 0.5 0.625 0.75 0.875)
    nnodes=${NNODES:-16}
    max_steps_divisor=512
    interval_divisor=10
    use_streaming=true
    num_epochs=""
    ;;
  3m)
    total_nums=(3000000)
    prop1_values=(0.125 0.25 0.375 0.5 0.625 0.75 0.875)
    nnodes=${NNODES:-16}
    max_steps_divisor=512
    interval_divisor=15
    use_streaming=true
    num_epochs=""
    ;;
  *)
    echo "Unknown config: ${CONFIG}" >&2
    usage
    exit 1
    ;;
esac

echo "Using config: ${CONFIG}"
echo "Seeds: ${seeds[*]}"

for total_num in "${total_nums[@]}"; do
  for prop1 in "${prop1_values[@]}"; do
    num_dataset1=$(python -c "print(int(${total_num}*${prop1}))")
    num_dataset2=$(python -c "print(int(${total_num}-${num_dataset1}))")

    dataset_1="${dataset1_base}_${num_dataset1}"
    dataset_2="${dataset2_base}_${num_dataset2}"

    max_steps=$(python -c "import math; print(math.ceil(${total_num}*3/${max_steps_divisor}))")
    interval=$(python -c "import math; print(max(1, int(${max_steps}//${interval_divisor})))")

    for seed in "${seeds[@]}"; do
      echo "total=${total_num}, prop1=${prop1}, seed=${seed}"
      echo "dataset1=${dataset_1}, dataset2=${dataset_2}"
      echo "max_steps=${max_steps}, interval=${interval}"

      output_dir="saves/recipe_exp1/${model_name}/${total_num}/${model_name}_${dataset_1}_${dataset_2}_${lr}_seed${seed}"

      torchrun_args=(
        --nproc_per_node=8
        --nnodes="${nnodes}"
        --rdzv_id="$SLURM_JOB_ID"
        --rdzv_backend=c10d
        --rdzv_endpoint="${head_node_ip}:29500"
        src/train.py
        --model_name_or_path "checkpoints/${model_name}"
        --stage sft
        --do_train
        --finetuning_type full
        --deepspeed "examples/deepspeed/ds_z3_config.json"
        --dataset "${dataset_1},${dataset_2}"
        --template llama3
        --cutoff_len 4096
        --overwrite_cache
        --output_dir "${output_dir}"
        --logging_steps "${interval}"
        --save_steps "${interval}"
        --eval_steps "${interval}"
        --plot_loss
        --overwrite_output_dir
        --per_device_train_batch_size 2
        --gradient_accumulation_steps 2
        --learning_rate "${lr}"
        --dispatch_batches False
        --lr_scheduler_type linear
        --warmup_ratio 0.03
        --bf16
        --ddp_timeout 180000000
        --per_device_eval_batch_size 4
        --eval_strategy steps
        --eval_dataset recipe_exp1_val
        --load_best_model_at_end True
        --save_total_limit 1
        --seed "${seed}"
      )

      if [[ "${use_streaming}" == "true" ]]; then
        torchrun_args+=(--streaming True)
      fi

      if [[ -n "${num_epochs}" ]]; then
        torchrun_args+=(--num_train_epochs "${num_epochs}")
      else
        torchrun_args+=(--max_steps "${max_steps}")
      fi

      srun torchrun "${torchrun_args[@]}" \
        >> "printout/output_file/output_${SLURM_JOB_ID}_${current_time}_${seed}.out" \
        2>> "printout/error_file/error_${SLURM_JOB_ID}_${current_time}_${seed}.err"
    done
  done
done


