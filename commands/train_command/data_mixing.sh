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
Usage: data_mixing.sh <experiment> [variant]

Experiments:
  exp1 [default]             Llama3 8B streaming, domain/size sweeps.
  exp2 <variant>             Llama3 8B proportional mixes.
                              Variants: 5m, 20m, 200m, 5m-optim, 20m-optim, 200m-optim
  exp3 <variant>             OLMo streaming experiments.
                              Variants: 660k, 5m, 20m, 200m
  orca <variant>             ORCA data mixes across models.
                              Variants: llama3-8b, llama3-70b, qwen2.5-1.5b, qwen2.5-32b
  tulu3 <variant>            Tulu3 mixtures.
                              Variants: llama3-8b, llama3-70b, llama3-70b-submod, qwen2.5-32b

Override defaults with environment variables, e.g.:
  sbatch --nodes=8 data_mixing.sh exp2 20m
EOF
}

[[ $# -ge 1 ]] || { usage; exit 1; }

experiment="$1"
shift || true

function slurm_head_info() {
  local head_node
  head_node=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
  local head_ip
  head_ip=$(getent hosts "$head_node" | awk '{ print $1 }')
  echo "${head_node};${head_ip}"
}

function ensure_output_dirs() {
  mkdir -p printout/output_file printout/error_file
}

function run_exp1() {
  local domains_input=(${DOMAINS_OVERRIDE:-instr math code})
  local sizes_input=(${SIZE_OVERRIDE:-one half third double triple})
  local seeds_input=(${SEEDS:-42})
  local base_domains=("instr" "math" "code")

  init_environment "cuda/12.4"
  ensure_output_dirs
  nvidia-smi

  local head_info
  head_info=$(slurm_head_info)
  local head_node="${head_info%;*}"
  local head_node_ip="${head_info#*;}"
  echo "Head node: ${head_node} (${head_node_ip})"

  local current_time
  current_time=$(timestamp)

  local model_name=${MODEL_NAME:-"Llama-3.1-8B"}
  local base_token=${BASE_TOKEN:-660000}
  local lr=${LR:-1.0e-5}
  local exp_tag=${EXP_TAG:-exp1}

  for domain in "${domains_input[@]}"; do
    for size in "${sizes_input[@]}"; do
      dataset1="${domain}_${size}"

      other_domains=()
      for d in "${base_domains[@]}"; do
        if [[ "$d" != "$domain" ]]; then
          other_domains+=("$d")
        fi
      done

      dataset2="${other_domains[0]}_one"
      dataset3="${other_domains[1]}_one"

      for seed in "${seeds_input[@]}"; do
        echo "exp1 domain=${domain}, size=${size}, seed=${seed}"

        output_dir="saves/data_mixing/${exp_tag}/${model_name}/${base_token}/${domain}/${model_name}_${dataset1}_${dataset2}_${dataset3}_${lr}_seed${seed}"

        srun torchrun \
          --nproc_per_node=8 \
          --nnodes="${NNODES:-8}" \
          --rdzv_id="$SLURM_JOB_ID" \
          --rdzv_backend=c10d \
          --rdzv_endpoint="${head_node_ip}:29500" \
          src/train.py \
          --model_name_or_path "checkpoints/${model_name}" \
          --stage sft \
          --do_train \
          --finetuning_type full \
          --deepspeed "examples/deepspeed/ds_z3_config.json" \
          --dataset "${dataset1},${dataset2},${dataset3}" \
          --template llama3 \
          --cutoff_len 4096 \
          --overwrite_cache \
          --output_dir "${output_dir}" \
          --logging_steps 5 \
          --save_steps 5 \
          --plot_loss \
          --overwrite_output_dir \
          --per_device_train_batch_size 1 \
          --gradient_accumulation_steps 2 \
          --learning_rate "${lr}" \
          --dispatch_batches False \
          --lr_scheduler_type cosine \
          --warmup_ratio 0.05 \
          --bf16 \
          --ddp_timeout 180000000 \
          --per_device_eval_batch_size 2 \
          --eval_strategy steps \
          --eval_dataset instr_val,math_val,code_val \
          --eval_steps 5 \
          --load_best_model_at_end True \
          --save_total_limit 1 \
          --include_num_input_tokens_seen True \
          --seed "${seed}" \
          --streaming True \
          --max_steps 100 \
          >> "printout/output_file/output_${SLURM_JOB_ID}_${current_time}_${seed}.out" \
          2>> "printout/error_file/error_${SLURM_JOB_ID}_${current_time}_${seed}.err"
      done
    done
  done
}

function run_exp2() {
  local variant="${1:-200m}"
  shift || true

  init_environment "cuda/12.4"
  ensure_output_dirs
  nvidia-smi

  local head_info
  head_info=$(slurm_head_info)
  local head_node="${head_info%;*}"
  local head_node_ip="${head_info#*;}"
  echo "Head node: ${head_node} (${head_node_ip})"

  local current_time
  current_time=$(timestamp)

  local model_name=${MODEL_NAME:-"Llama-3.1-8B"}
  local lr=${LR:-1.0e-5}
  local seeds_input=(${SEEDS:-42})

  local base_token nnodes size_pairs sizes eval_dataset eval_steps logging_steps max_steps gradient_steps batch_size scheduler warmup_ratio per_device_eval
  local dataset_mode

  case "${variant}" in
    5m)
      base_token=5000000
      nnodes=${NNODES:-8}
      size_pairs=("0.5 0.375" "0.375 0.5")
      sizes=(0.125 0.25 0.375 0.5 0.625 0.75)
      logging_steps=10
      max_steps=500
      gradient_steps=2
      batch_size=1
      scheduler=linear
      warmup_ratio=0.0
      per_device_eval=2
      eval_dataset="5000000_exp2_val"
      eval_steps=10
      dataset_mode="proportional"
      ;;
    20m)
      base_token=20000000
      nnodes=${NNODES:-16}
      size_pairs=("0.125 0.125" "0.625 0.125" "0.625 0.25")
      sizes=(0.125 0.25 0.375 0.5 0.625 0.75)
      logging_steps=10
      max_steps=400
      gradient_steps=2
      batch_size=1
      scheduler=linear
      warmup_ratio=0.0
      per_device_eval=1
      eval_dataset="5000000_exp2_val"
      eval_steps=10
      dataset_mode="proportional"
      ;;
    200m)
      base_token=200000000
      nnodes=${NNODES:-16}
      size_pairs=("0.5 0.375" "0.375 0.5" "0.625 0.25" "0.625 0.125")
      sizes=(0.125 0.25 0.375 0.5 0.625 0.75)
      logging_steps=50
      max_steps=3000
      gradient_steps=2
      batch_size=1
      scheduler=linear
      warmup_ratio=0.0
      per_device_eval=4
      eval_dataset="5000000_exp2_val"
      eval_steps=50
      dataset_mode="proportional"
      ;;
    5m-optim)
      base_token=5000000
      nnodes=${NNODES:-8}
      logging_steps=10
      max_steps=150
      gradient_steps=4
      batch_size=1
      scheduler=cosine
      warmup_ratio=0.05
      per_device_eval=4
      eval_dataset="5000000_exp2_val"
      eval_steps=10
      dataset_mode="optim"
      ;;
    20m-optim)
      base_token=20000000
      nnodes=${NNODES:-8}
      logging_steps=10
      max_steps=500
      gradient_steps=4
      batch_size=1
      scheduler=cosine
      warmup_ratio=0.05
      per_device_eval=4
      eval_dataset="5000000_exp2_val"
      eval_steps=10
      dataset_mode="optim"
      ;;
    200m-optim)
      base_token=200000000
      nnodes=${NNODES:-16}
      logging_steps=40
      max_steps=1500
      gradient_steps=4
      batch_size=1
      scheduler=cosine
      warmup_ratio=0.05
      per_device_eval=4
      eval_dataset="5000000_exp2_val"
      eval_steps=40
      dataset_mode="optim"
      ;;
    *)
      echo "Unknown exp2 variant: ${variant}" >&2
      usage
      exit 1
      ;;
  esac

  local domains=("instr" "math" "code")

  if [[ "${dataset_mode}" == "optim" ]]; then
    local dataset_instr="${base_token}_${domains[0]}_${base_token}_${model_name}_${domains[0]}_optim"
    local dataset_math="${base_token}_${domains[1]}_${base_token}_${model_name}_${domains[1]}_optim"
    local dataset_code="${base_token}_${domains[2]}_${base_token}_${model_name}_${domains[2]}_optim"

    for seed in "${seeds_input[@]}"; do
      echo "exp2 optim base=${base_token}, seed=${seed}"

      output_dir="saves/data_mixing/exp2/${model_name}/${base_token}/optim/${model_name}_${dataset_instr}_${dataset_math}_${dataset_code}_${lr}_seed${seed}"

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
        --dataset "${dataset_instr},${dataset_math},${dataset_code}" \
        --template llama3 \
        --cutoff_len 4096 \
        --overwrite_cache \
        --output_dir "${output_dir}" \
        --logging_steps "${logging_steps}" \
        --save_steps "${logging_steps}" \
        --plot_loss \
        --overwrite_output_dir \
        --per_device_train_batch_size "${batch_size}" \
        --gradient_accumulation_steps "${gradient_steps}" \
        --learning_rate "${lr}" \
        --dispatch_batches False \
        --lr_scheduler_type "${scheduler}" \
        --warmup_ratio "${warmup_ratio}" \
        --bf16 \
        --ddp_timeout 180000 \
        --per_device_eval_batch_size "${per_device_eval}" \
        --eval_strategy steps \
        --eval_dataset "${eval_dataset}" \
        --eval_steps "${eval_steps}" \
        --load_best_model_at_end True \
        --save_total_limit 1 \
        --include_num_input_tokens_seen True \
        --seed "${seed}" \
        --streaming True \
        --max_steps "${max_steps}" \
        >> "printout/output_file/output_${SLURM_JOB_ID}_${current_time}_${seed}.out" \
        2>> "printout/error_file/error_${SLURM_JOB_ID}_${current_time}_${seed}.err"
    done
    return
  fi

  for pair in "${size_pairs[@]}"; do
    read -r size1 size2 <<< "${pair}"
    for size3 in "${sizes[@]}"; do
      is_one=$(python - "$size1" "$size2" "$size3" <<'PY'
import decimal, sys
decimal.getcontext().prec = 6
s = sum(decimal.Decimal(x) for x in sys.argv[1:])
print(1 if abs(s - decimal.Decimal("1")) < decimal.Decimal("1e-6") else 0)
PY
)
      if [[ "${is_one}" != "1" ]]; then
        continue
      fi

      dataset_instr="${base_token}_${domains[0]}_${size1}"
      dataset_math="${base_token}_${domains[1]}_${size2}"
      dataset_code="${base_token}_${domains[2]}_${size3}"

      for seed in "${seeds_input[@]}"; do
        echo "exp2 base=${base_token}, sizes=(${size1},${size2},${size3}), seed=${seed}"

        output_dir="saves/data_mixing/exp2/${model_name}/${base_token}/${domains[0]}-${domains[1]}-${domains[2]}/${model_name}_${dataset_instr}_${dataset_math}_${dataset_code}_${lr}_seed${seed}"

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
          --dataset "${dataset_instr},${dataset_math},${dataset_code}" \
          --template llama3 \
          --cutoff_len 4096 \
          --overwrite_cache \
          --output_dir "${output_dir}" \
          --logging_steps "${logging_steps}" \
          --save_steps "${logging_steps}" \
          --plot_loss \
          --overwrite_output_dir \
          --per_device_train_batch_size "${batch_size}" \
          --gradient_accumulation_steps "${gradient_steps}" \
          --learning_rate "${lr}" \
          --dispatch_batches False \
          --lr_scheduler_type "${scheduler}" \
          --warmup_ratio "${warmup_ratio}" \
          --bf16 \
          --ddp_timeout 180000 \
          --per_device_eval_batch_size "${per_device_eval}" \
          --eval_strategy steps \
          --eval_dataset "${eval_dataset}" \
          --eval_steps "${eval_steps}" \
          --load_best_model_at_end True \
          --save_total_limit 1 \
          --include_num_input_tokens_seen True \
          --seed "${seed}" \
          --streaming True \
          --max_steps "${max_steps}" \
          >> "printout/output_file/output_${SLURM_JOB_ID}_${current_time}_${seed}.out" \
          2>> "printout/error_file/error_${SLURM_JOB_ID}_${current_time}_${seed}.err"
      done
    done
  done
}

function run_exp3() {
  local variant="${1:-660k}"
  shift || true

  init_environment "cuda/12.4" "${DEFAULT_CONDA_HOME}" "${CONDA_ENV:-olmo}"
  ensure_output_dirs
  nvidia-smi

  export DISABLE_VERSION_CHECK=1

  local head_info
  head_info=$(slurm_head_info)
  local head_node="${head_info%;*}"
  local head_node_ip="${head_info#*;}"
  echo "Head node: ${head_node} (${head_node_ip})"

  local current_time
  current_time=$(timestamp)

  local model_name=${MODEL_NAME:-"OLMo-2-7B"}
  local lr=${LR:-1.0e-5}
  local seeds_input=(${SEEDS:-42})

  case "${variant}" in
    660k)
      local base_token=660000
      local domains=("instr" "math" "code")
      local sizes=("one" "half" "third" "double" "triple")
      local nnodes=${NNODES:-8}
      local logging_steps=5
      local max_steps=100
      local batch_size=1
      local grad_steps=2
      local scheduler=cosine
      local warmup=0.05
      local eval_steps=5
      local eval_dataset="5000000_exp2_val"

      for domain in "${domains[@]}"; do
        for size in "${sizes[@]}"; do
          dataset1="${domain}_${size}"
          other_domains=()
          for d in "${domains[@]}"; do
            if [[ "$d" != "$domain" ]]; then
              other_domains+=("$d")
            fi
          done
          dataset2="${other_domains[0]}_one"
          dataset3="${other_domains[1]}_one"

          for seed in "${seeds_input[@]}"; do
            echo "exp3-660k domain=${domain}, size=${size}, seed=${seed}"
            output_dir="saves/data_mixing/exp3/${model_name}/${base_token}/${domain}/${model_name}_${dataset1}_${dataset2}_${dataset3}_${lr}_seed${seed}"

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
              --dataset "${dataset1},${dataset2},${dataset3}" \
              --template olmo \
              --cutoff_len 4096 \
              --overwrite_cache \
              --output_dir "${output_dir}" \
              --logging_steps "${logging_steps}" \
              --save_steps "${logging_steps}" \
              --plot_loss \
              --overwrite_output_dir \
              --per_device_train_batch_size "${batch_size}" \
              --gradient_accumulation_steps "${grad_steps}" \
              --learning_rate "${lr}" \
              --dispatch_batches False \
              --lr_scheduler_type "${scheduler}" \
              --warmup_ratio "${warmup}" \
              --bf16 \
              --ddp_timeout 180000000 \
              --per_device_eval_batch_size 4 \
              --eval_strategy steps \
              --eval_dataset "${eval_dataset}" \
              --eval_steps "${eval_steps}" \
              --load_best_model_at_end True \
              --save_total_limit 1 \
              --include_num_input_tokens_seen True \
              --seed "${seed}" \
              --streaming True \
              --max_steps "${max_steps}" \
              >> "printout/output_file/output_${SLURM_JOB_ID}_${current_time}_${seed}.out" \
              2>> "printout/error_file/error_${SLURM_JOB_ID}_${current_time}_${seed}.err"
          done
        done
      done
      ;;
    5m|20m|200m)
      local base_token
      local nnodes
      local logging_steps
      local max_steps
      local batch_size
      local grad_steps=2
      local scheduler=constant
      local warmup=0.0
      local eval_dataset="5000000_exp2_val"
      local eval_steps

      case "${variant}" in
        5m)
          base_token=5000000
          nnodes=${NNODES:-8}
          logging_steps=5
          max_steps=100
          batch_size=2
          eval_steps=5
          ;;
        20m)
          base_token=20000000
          nnodes=${NNODES:-8}
          logging_steps=10
          max_steps=400
          batch_size=2
          eval_steps=5
          ;;
        200m)
          base_token=200000000
          nnodes=${NNODES:-16}
          logging_steps=40
          max_steps=2000
          batch_size=1
          eval_steps=40
          ;;
      esac

      local domains=("instr" "math" "code")
      local size_grid=(0.125 0.25 0.375 0.5 0.625 0.75)

      for size1 in "${size_grid[@]}"; do
        for size2 in "${size_grid[@]}"; do
      for size3 in "${size_grid[@]}"; do
        is_one=$(python - "$size1" "$size2" "$size3" <<'PY'
import decimal, sys
decimal.getcontext().prec = 6
s = sum(decimal.Decimal(x) for x in sys.argv[1:])
print(1 if abs(s - decimal.Decimal("1")) < decimal.Decimal("1e-6") else 0)
PY
)
        if [[ "${is_one}" != "1" ]]; then
              continue
            fi

            dataset_instr="${base_token}_${domains[0]}_${size1}"
            dataset_math="${base_token}_${domains[1]}_${size2}"
            dataset_code="${base_token}_${domains[2]}_${size3}"

            for seed in "${seeds_input[@]}"; do
              echo "exp3-${variant} sizes=(${size1},${size2},${size3}), seed=${seed}"

              output_dir="saves/data_mixing/exp3/${model_name}/${base_token}/${domains[0]}-${domains[1]}-${domains[2]}/${model_name}_${dataset_instr}_${dataset_math}_${dataset_code}_${lr}_seed${seed}"

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
                --dataset "${dataset_instr},${dataset_math},${dataset_code}" \
                --template olmo \
                --cutoff_len 4096 \
                --overwrite_cache \
                --output_dir "${output_dir}" \
                --logging_steps "${logging_steps}" \
                --save_steps "${logging_steps}" \
                --plot_loss \
                --overwrite_output_dir \
                --per_device_train_batch_size "${batch_size}" \
                --gradient_accumulation_steps "${grad_steps}" \
                --learning_rate "${lr}" \
                --dispatch_batches False \
                --lr_scheduler_type "${scheduler}" \
                --warmup_ratio "${warmup}" \
                --bf16 \
                --ddp_timeout 180000 \
                --per_device_eval_batch_size 4 \
                --eval_strategy steps \
                --eval_dataset "${eval_dataset}" \
                --eval_steps "${eval_steps}" \
                --load_best_model_at_end True \
                --save_total_limit 1 \
                --include_num_input_tokens_seen True \
                --seed "${seed}" \
                --streaming True \
                --max_steps "${max_steps}" \
                >> "printout/output_file/output_${SLURM_JOB_ID}_${current_time}_${seed}.out" \
                2>> "printout/error_file/error_${SLURM_JOB_ID}_${current_time}_${seed}.err"
            done
          done
        done
      done
      ;;
    *)
      echo "Unknown exp3 variant: ${variant}" >&2
      usage
      exit 1
      ;;
  esac
}

function run_orca() {
  local variant="${1:-llama3-8b}"
  shift || true

  init_environment "cuda/12.4"
  ensure_output_dirs
  nvidia-smi

  local head_info
  head_info=$(slurm_head_info)
  local head_node="${head_info%;*}"
  local head_node_ip="${head_info#*;}"
  echo "Head node: ${head_node} (${head_node_ip})"

  local current_time
  current_time=$(timestamp)

  local seeds_input=(${SEEDS:-42})
  local exp_name=${EXP_NAME:-orca}

  case "${variant}" in
    llama3-8b)
      local model_name=${MODEL_NAME:-"Llama-3.1-8B"}
      local weight_type=${WEIGHT_TYPE:-ours}
      local dataset_name=${DATASET_NAME:-"orca_ours"}
      local lr=${LR:-1.0e-5}
      local nnodes=${NNODES:-8}
      local logging_steps=${LOGGING_STEPS:-100}
      local max_steps=${MAX_STEPS:-5000}
      local grad_steps=${GRAD_STEPS:-4}
      local batch_size=${BATCH_SIZE:-1}
      local eval_steps=${EVAL_STEPS:-100}
      local per_device_eval=${PER_DEVICE_EVAL:-2}

      for seed in "${seeds_input[@]}"; do
        echo "orca ${variant} seed=${seed}"
        output_dir="saves/data_mixing/${exp_name}/${model_name}/${weight_type}/${model_name}_${dataset_name}_${lr}_seed${seed}"

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
          --output_dir "${output_dir}" \
          --logging_steps "${logging_steps}" \
          --save_steps "${logging_steps}" \
          --plot_loss \
          --overwrite_output_dir \
          --per_device_train_batch_size "${batch_size}" \
          --gradient_accumulation_steps "${grad_steps}" \
          --learning_rate "${lr}" \
          --dispatch_batches False \
          --lr_scheduler_type linear \
          --warmup_ratio 0.05 \
          --bf16 \
          --ddp_timeout 180000 \
          --per_device_eval_batch_size "${per_device_eval}" \
          --eval_strategy steps \
          --eval_dataset orca_t0_val,orca_cot_val,orca_flan_val,orca_niv_val \
          --eval_steps "${eval_steps}" \
          --load_best_model_at_end True \
          --save_total_limit 1 \
          --include_num_input_tokens_seen True \
          --seed "${seed}" \
          --streaming True \
          --max_steps "${max_steps}" \
          >> "printout/output_file/output_${SLURM_JOB_ID}_${current_time}_${seed}.out" \
          2>> "printout/error_file/error_${SLURM_JOB_ID}_${current_time}_${seed}.err"
      done
      ;;
    llama3-70b)
      local model_name=${MODEL_NAME:-"Llama-3.1-70B"}
      local weight_type=${WEIGHT_TYPE:-ours}
      local dataset_name=${DATASET_NAME:-"orca_ours"}
      local lr=${LR:-1.0e-6}
      local nnodes=${NNODES:-16}
      local logging_steps=${LOGGING_STEPS:-100}
      local max_steps=${MAX_STEPS:-4500}
      local grad_steps=${GRAD_STEPS:-2}
      local batch_size=${BATCH_SIZE:-1}
      local eval_steps=${EVAL_STEPS:-100}
      local per_device_eval=${PER_DEVICE_EVAL:-1}

      for seed in "${seeds_input[@]}"; do
        echo "orca ${variant} seed=${seed}"
        output_dir="saves/data_mixing/${exp_name}/${model_name}/${weight_type}/${model_name}_${dataset_name}_${lr}_seed${seed}"

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
          --output_dir "${output_dir}" \
          --logging_steps "${logging_steps}" \
          --save_steps "${logging_steps}" \
          --plot_loss \
          --overwrite_output_dir \
          --per_device_train_batch_size "${batch_size}" \
          --gradient_accumulation_steps "${grad_steps}" \
          --learning_rate "${lr}" \
          --dispatch_batches False \
          --lr_scheduler_type linear \
          --warmup_ratio 0.05 \
          --bf16 \
          --ddp_timeout 180000 \
          --per_device_eval_batch_size "${per_device_eval}" \
          --eval_strategy steps \
          --eval_dataset orca_t0_val,orca_cot_val,orca_flan_val,orca_niv_val \
          --eval_steps "${eval_steps}" \
          --load_best_model_at_end True \
          --save_total_limit 1 \
          --include_num_input_tokens_seen True \
          --seed "${seed}" \
          --streaming True \
          --max_steps "${max_steps}" \
          >> "printout/output_file/output_${SLURM_JOB_ID}_${current_time}_${seed}.out" \
          2>> "printout/error_file/error_${SLURM_JOB_ID}_${current_time}_${seed}.err"
      done
      ;;
    llama3-70b-resume)
      local model_name=${MODEL_NAME:-"Llama-3.1-70B"}
      local weight_type=${WEIGHT_TYPE:-submodular}
      local dataset_name=${DATASET_NAME:-"orca_submodular"}
      local lr=${LR:-1.0e-6}
      local nnodes=${NNODES:-16}
      local logging_steps=${LOGGING_STEPS:-50}
      local max_steps=${MAX_STEPS:-3500}
      local grad_steps=${GRAD_STEPS:-2}
      local batch_size=${BATCH_SIZE:-1}
      local eval_steps=${EVAL_STEPS:-50}
      local per_device_eval=${PER_DEVICE_EVAL:-2}
      local resume_default="saves/data_mixing/orca/Llama-3.1-70B/submodular/Llama-3.1-70B_orca_submodular_1.0e-6_seed42/checkpoint-2300"
      local resume_path=${RESUME:-${resume_default}}

      for seed in "${seeds_input[@]}"; do
        echo "orca ${variant} seed=${seed} resume=${resume_path}"
        output_dir="saves/data_mixing/${exp_name}/${model_name}/${weight_type}/${model_name}_${dataset_name}_${lr}_seed${seed}"

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
          --output_dir "${output_dir}" \
          --logging_steps "${logging_steps}" \
          --save_steps "${logging_steps}" \
          --plot_loss \
          --overwrite_output_dir \
          --per_device_train_batch_size "${batch_size}" \
          --gradient_accumulation_steps "${grad_steps}" \
          --learning_rate "${lr}" \
          --dispatch_batches False \
          --lr_scheduler_type cosine \
          --warmup_ratio 0.05 \
          --bf16 \
          --ddp_timeout 180000 \
          --per_device_eval_batch_size "${per_device_eval}" \
          --eval_strategy steps \
          --eval_dataset orca_t0_val,orca_cot_val,orca_flan_val,orca_niv_val \
          --eval_steps "${eval_steps}" \
          --load_best_model_at_end True \
          --save_total_limit 1 \
          --include_num_input_tokens_seen True \
          --seed "${seed}" \
          --streaming True \
          --resume_from_checkpoint "${resume_path}" \
          --max_steps "${max_steps}" \
          >> "printout/output_file/output_${SLURM_JOB_ID}_${current_time}_${seed}.out" \
          2>> "printout/error_file/error_${SLURM_JOB_ID}_${current_time}_${seed}.err"
      done
      ;;
    llama3-70b-660k)
      local model_name=${MODEL_NAME:-"Llama-3.1-70B"}
      local lr=${LR:-1.0e-6}
      local nnodes=${NNODES:-16}
      local logging_steps=${LOGGING_STEPS:-5}
      local max_steps=${MAX_STEPS:-80}
      local grad_steps=${GRAD_STEPS:-2}
      local batch_size=${BATCH_SIZE:-1}
      local eval_steps=${EVAL_STEPS:-5}
      local per_device_eval=${PER_DEVICE_EVAL:-1}
      local base_token=${BASE_TOKEN:-660000}
      local base_domains=("t0" "cot" "flan" "niv")
      local focus_domains=(${FOCUS_DOMAINS:-niv})
      local size_list=(${SIZE_OVERRIDE:-triple})

      for domain in "${focus_domains[@]}"; do
        for size in "${size_list[@]}"; do
          other_domains=()
          for d in "${base_domains[@]}"; do
            if [[ "$d" != "$domain" ]]; then
              other_domains+=("$d")
            fi
          done

          dataset1="${base_token}_${domain}_${size}"
          dataset2="${base_token}_${other_domains[0]}_one"
          dataset3="${base_token}_${other_domains[1]}_one"
          dataset4="${base_token}_${other_domains[2]}_one"

          for seed in "${seeds_input[@]}"; do
            echo "orca ${variant} domain=${domain}, size=${size}, seed=${seed}"
            output_dir="saves/data_mixing/${exp_name}/${model_name}/${base_token}/${domain}/${model_name}_${dataset1}_${dataset2}_${dataset3}_${dataset4}_${lr}_seed${seed}"

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
              --dataset "${dataset1},${dataset2},${dataset3},${dataset4}" \
              --template llama3 \
              --cutoff_len 4096 \
              --overwrite_cache \
              --output_dir "${output_dir}" \
              --logging_steps "${logging_steps}" \
              --save_steps "${logging_steps}" \
              --plot_loss \
              --overwrite_output_dir \
              --per_device_train_batch_size "${batch_size}" \
              --gradient_accumulation_steps "${grad_steps}" \
              --learning_rate "${lr}" \
              --dispatch_batches False \
              --lr_scheduler_type cosine \
              --warmup_ratio 0.05 \
              --bf16 \
              --ddp_timeout 180000 \
              --per_device_eval_batch_size "${per_device_eval}" \
              --eval_strategy steps \
              --eval_dataset orca_t0_val,orca_cot_val,orca_flan_val,orca_niv_val \
              --eval_steps "${eval_steps}" \
              --load_best_model_at_end True \
              --save_total_limit 1 \
              --include_num_input_tokens_seen True \
              --seed "${seed}" \
              --streaming True \
              --max_steps "${max_steps}" \
              >> "printout/output_file/output_${SLURM_JOB_ID}_${current_time}_${seed}.out" \
              2>> "printout/error_file/error_${SLURM_JOB_ID}_${current_time}_${seed}.err"
          done
        done
      done
      ;;
    qwen2.5-1.5b)
      local model_name=${MODEL_NAME:-"Qwen2.5-1.5B"}
      local lr=${LR:-2.0e-5}
      local nnodes=${NNODES:-8}
      local logging_steps=${LOGGING_STEPS:-10}
      local max_steps=${MAX_STEPS:-150}
      local grad_steps=${GRAD_STEPS:-4}
      local batch_size=${BATCH_SIZE:-2}
      local eval_steps=${EVAL_STEPS:-10}
      local per_device_eval=${PER_DEVICE_EVAL:-1}
      local base_token=${BASE_TOKEN:-660000}
      local domains=("t0" "cot" "flan" "niv")
      local sizes=("one" "half" "third" "double" "triple")

      for domain in "${domains[@]}"; do
        for size in "${sizes[@]}"; do
          other_domains=()
          for d in "${domains[@]}"; do
            if [[ "$d" != "$domain" ]]; then
              other_domains+=("$d")
            fi
          done

          dataset1="${base_token}_${domain}_${size}"
          dataset2="${base_token}_${other_domains[0]}_one"
          dataset3="${base_token}_${other_domains[1]}_one"
          dataset4="${base_token}_${other_domains[2]}_one"

          for seed in "${seeds_input[@]}"; do
            echo "orca ${variant} domain=${domain}, size=${size}, seed=${seed}"
            output_dir="saves/data_mixing/${exp_name}/${model_name}/${base_token}/${domain}/${model_name}_${dataset1}_${dataset2}_${dataset3}_${dataset4}_${lr}_seed${seed}"

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
              --dataset "${dataset1},${dataset2},${dataset3},${dataset4}" \
              --template qwen \
              --cutoff_len 4096 \
              --overwrite_cache \
              --output_dir "${output_dir}" \
              --logging_steps "${logging_steps}" \
              --save_steps "${logging_steps}" \
              --plot_loss \
              --overwrite_output_dir \
              --per_device_train_batch_size "${batch_size}" \
              --gradient_accumulation_steps "${grad_steps}" \
              --learning_rate "${lr}" \
              --dispatch_batches False \
              --lr_scheduler_type cosine \
              --warmup_ratio 0.05 \
              --bf16 \
              --ddp_timeout 180000 \
              --per_device_eval_batch_size "${per_device_eval}" \
              --eval_strategy steps \
              --eval_dataset orca_t0_val,orca_cot_val,orca_flan_val,orca_niv_val \
              --eval_steps "${eval_steps}" \
              --load_best_model_at_end True \
              --save_total_limit 1 \
              --include_num_input_tokens_seen True \
              --seed "${seed}" \
              --streaming True \
              --max_steps "${max_steps}" \
              >> "printout/output_file/output_${SLURM_JOB_ID}_${current_time}_${seed}.out" \
              2>> "printout/error_file/error_${SLURM_JOB_ID}_${current_time}_${seed}.err"
          done
        done
      done
      ;;
    qwen2.5-32b)
      export CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING:-1}
      local model_name=${MODEL_NAME:-"Qwen2.5-32B"}
      local weight_type=${WEIGHT_TYPE:-ours}
      local dataset_name=${DATASET_NAME:-"orca_Qwen_ours"}
      local lr=${LR:-2.0e-6}
      local nnodes=${NNODES:-8}
      local logging_steps=${LOGGING_STEPS:-50}
      local max_steps=${MAX_STEPS:-4500}
      local grad_steps=${GRAD_STEPS:-4}
      local batch_size=${BATCH_SIZE:-1}
      local eval_steps=${EVAL_STEPS:-50}
      local per_device_eval=${PER_DEVICE_EVAL:-1}

      for seed in "${seeds_input[@]}"; do
        echo "orca ${variant} seed=${seed}"
        output_dir="saves/data_mixing/${exp_name}/${model_name}/${weight_type}/${model_name}_${dataset_name}_${lr}_seed${seed}"

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
          --template qwen \
          --cutoff_len 4096 \
          --overwrite_cache \
          --output_dir "${output_dir}" \
          --logging_steps "${logging_steps}" \
          --save_steps "${logging_steps}" \
          --plot_loss \
          --overwrite_output_dir \
          --per_device_train_batch_size "${batch_size}" \
          --gradient_accumulation_steps "${grad_steps}" \
          --learning_rate "${lr}" \
          --dispatch_batches False \
          --lr_scheduler_type linear \
          --warmup_ratio 0.05 \
          --bf16 \
          --ddp_timeout 180000 \
          --per_device_eval_batch_size "${per_device_eval}" \
          --eval_strategy steps \
          --eval_dataset orca_t0_val,orca_cot_val,orca_flan_val,orca_niv_val \
          --eval_steps "${eval_steps}" \
          --load_best_model_at_end True \
          --save_total_limit 1 \
          --include_num_input_tokens_seen True \
          --seed "${seed}" \
          --streaming True \
          --max_steps "${max_steps}" \
          >> "printout/output_file/output_${SLURM_JOB_ID}_${current_time}_${seed}.out" \
          2>> "printout/error_file/error_${SLURM_JOB_ID}_${current_time}_${seed}.err"
      done
      ;;
    *)
      echo "Unknown orca variant: ${variant}" >&2
      usage
      exit 1
      ;;
  esac
}

function run_tulu3() {
  local variant="${1:-llama3-8b}"
  shift || true

  init_environment "cuda/12.4"
  ensure_output_dirs
  nvidia-smi

  local head_info
  head_info=$(slurm_head_info)
  local head_node="${head_info%;*}"
  local head_node_ip="${head_info#*;}"
  echo "Head node: ${head_node} (${head_node_ip})"

  local current_time
  current_time=$(timestamp)

  local seeds_input=(${SEEDS:-42})
  local exp_name=${EXP_NAME:-tulu3}
  local eval_dataset_default="tulu3_general_val,tulu3_knowledge_recall_val,tulu3_math_val,tulu3_code_val,tulu3_safety_val,tulu3_precise_IF_val"

  case "${variant}" in
    llama3-8b)
      local model_name=${MODEL_NAME:-"Llama-3.1-8B"}
      local weight_type=${WEIGHT_TYPE:-ours}
      local dataset_name=${DATASET_NAME:-"tulu3_ours"}
      local lr=${LR:-1.0e-5}
      local nnodes=${NNODES:-8}
      local logging_steps=${LOGGING_STEPS:-100}
      local max_steps=${MAX_STEPS:-5000}
      local grad_steps=${GRAD_STEPS:-4}
      local batch_size=${BATCH_SIZE:-1}
      local eval_steps=${EVAL_STEPS:-100}
      local per_device_eval=${PER_DEVICE_EVAL:-1}

      for seed in "${seeds_input[@]}"; do
        echo "tulu3 ${variant} seed=${seed}"
        output_dir="saves/data_mixing/${exp_name}/${model_name}/${weight_type}/${model_name}_${dataset_name}_${lr}_seed${seed}"

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
          --output_dir "${output_dir}" \
          --logging_steps "${logging_steps}" \
          --save_steps "${logging_steps}" \
          --plot_loss \
          --overwrite_output_dir \
          --per_device_train_batch_size "${batch_size}" \
          --gradient_accumulation_steps "${grad_steps}" \
          --learning_rate "${lr}" \
          --dispatch_batches False \
          --lr_scheduler_type linear \
          --warmup_ratio 0.05 \
          --bf16 \
          --ddp_timeout 180000 \
          --per_device_eval_batch_size "${per_device_eval}" \
          --eval_strategy steps \
          --eval_dataset "${eval_dataset_default}" \
          --eval_steps "${eval_steps}" \
          --load_best_model_at_end True \
          --save_total_limit 1 \
          --include_num_input_tokens_seen True \
          --seed "${seed}" \
          --streaming True \
          --max_steps "${max_steps}" \
          >> "printout/output_file/output_${SLURM_JOB_ID}_${current_time}_${seed}.out" \
          2>> "printout/error_file/error_${SLURM_JOB_ID}_${current_time}_${seed}.err"
      done
      ;;
    llama3-70b)
      local model_name=${MODEL_NAME:-"Llama-3.1-70B"}
      local weight_type=${WEIGHT_TYPE:-ours}
      local dataset_name=${DATASET_NAME:-"tulu3_ours"}
      local lr=${LR:-1.0e-6}
      local nnodes=${NNODES:-16}
      local logging_steps=${LOGGING_STEPS:-50}
      local max_steps=${MAX_STEPS:-4500}
      local grad_steps=${GRAD_STEPS:-2}
      local batch_size=${BATCH_SIZE:-1}
      local eval_steps=${EVAL_STEPS:-50}
      local per_device_eval=${PER_DEVICE_EVAL:-1}

      for seed in "${seeds_input[@]}"; do
        echo "tulu3 ${variant} seed=${seed}"
        output_dir="saves/data_mixing/${exp_name}/${model_name}/${weight_type}/${model_name}_${dataset_name}_${lr}_seed${seed}"

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
          --output_dir "${output_dir}" \
          --logging_steps "${logging_steps}" \
          --save_steps "${logging_steps}" \
          --plot_loss \
          --overwrite_output_dir \
          --per_device_train_batch_size "${batch_size}" \
          --gradient_accumulation_steps "${grad_steps}" \
          --learning_rate "${lr}" \
          --dispatch_batches False \
          --lr_scheduler_type linear \
          --warmup_ratio 0.05 \
          --bf16 \
          --ddp_timeout 180000 \
          --per_device_eval_batch_size "${per_device_eval}" \
          --eval_strategy steps \
          --eval_dataset "${eval_dataset_default}" \
          --eval_steps "${eval_steps}" \
          --load_best_model_at_end True \
          --save_total_limit 1 \
          --include_num_input_tokens_seen True \
          --seed "${seed}" \
          --streaming True \
          --max_steps "${max_steps}" \
          >> "printout/output_file/output_${SLURM_JOB_ID}_${current_time}_${seed}.out" \
          2>> "printout/error_file/error_${SLURM_JOB_ID}_${current_time}_${seed}.err"
      done
      ;;
    llama3-70b-submod)
      local model_name=${MODEL_NAME:-"Llama-3.1-70B"}
      local weight_type=${WEIGHT_TYPE:-submodular}
      local dataset_name=${DATASET_NAME:-"tulu3_submodular"}
      local lr=${LR:-1.0e-6}
      local nnodes=${NNODES:-16}
      local logging_steps=${LOGGING_STEPS:-50}
      local max_steps=${MAX_STEPS:-4000}
      local grad_steps=${GRAD_STEPS:-2}
      local batch_size=${BATCH_SIZE:-1}
      local eval_steps=${EVAL_STEPS:-50}
      local per_device_eval=${PER_DEVICE_EVAL:-1}
      local resume_default="saves/data_mixing/tulu3/Llama-3.1-70B/submodular/Llama-3.1-70B_tulu3_submodular_1.0e-6_seed42/checkpoint-2550"
      local resume_path=${RESUME:-${resume_default}}

      for seed in "${seeds_input[@]}"; do
        echo "tulu3 ${variant} seed=${seed} resume=${resume_path}"
        output_dir="saves/data_mixing/${exp_name}/${model_name}/${weight_type}/${model_name}_${dataset_name}_${lr}_seed${seed}"

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
          --output_dir "${output_dir}" \
          --logging_steps "${logging_steps}" \
          --save_steps "${logging_steps}" \
          --plot_loss \
          --overwrite_output_dir \
          --per_device_train_batch_size "${batch_size}" \
          --gradient_accumulation_steps "${grad_steps}" \
          --learning_rate "${lr}" \
          --dispatch_batches False \
          --lr_scheduler_type constant \
          --warmup_ratio 0.0 \
          --bf16 \
          --ddp_timeout 180000 \
          --per_device_eval_batch_size "${per_device_eval}" \
          --eval_strategy steps \
          --eval_dataset "${eval_dataset_default}" \
          --eval_steps "${eval_steps}" \
          --load_best_model_at_end True \
          --save_total_limit 1 \
          --include_num_input_tokens_seen True \
          --seed "${seed}" \
          --streaming True \
          --resume_from_checkpoint "${resume_path}" \
          --max_steps "${max_steps}" \
          >> "printout/output_file/output_${SLURM_JOB_ID}_${current_time}_${seed}.out" \
          2>> "printout/error_file/error_${SLURM_JOB_ID}_${current_time}_${seed}.err"
      done
      ;;
    qwen2.5-32b)
      export CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING:-1}
      local model_name=${MODEL_NAME:-"Qwen2.5-32B"}
      local weight_type=${WEIGHT_TYPE:-ours}
      local dataset_name=${DATASET_NAME:-"tulu3_Qwen_ours"}
      local lr=${LR:-2.0e-6}
      local nnodes=${NNODES:-8}
      local logging_steps=${LOGGING_STEPS:-100}
      local max_steps=${MAX_STEPS:-5000}
      local grad_steps=${GRAD_STEPS:-4}
      local batch_size=${BATCH_SIZE:-1}
      local eval_steps=${EVAL_STEPS:-50}
      local per_device_eval=${PER_DEVICE_EVAL:-1}

      for seed in "${seeds_input[@]}"; do
        echo "tulu3 ${variant} seed=${seed}"
        output_dir="saves/data_mixing/${exp_name}/${model_name}/${weight_type}/${model_name}_${dataset_name}_${lr}_seed${seed}"

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
          --template qwen \
          --cutoff_len 4096 \
          --overwrite_cache \
          --output_dir "${output_dir}" \
          --logging_steps "${logging_steps}" \
          --save_steps "${logging_steps}" \
          --plot_loss \
          --overwrite_output_dir \
          --per_device_train_batch_size "${batch_size}" \
          --gradient_accumulation_steps "${grad_steps}" \
          --learning_rate "${lr}" \
          --dispatch_batches False \
          --lr_scheduler_type linear \
          --warmup_ratio 0.05 \
          --bf16 \
          --ddp_timeout 180000 \
          --per_device_eval_batch_size "${per_device_eval}" \
          --eval_strategy steps \
          --eval_dataset "${eval_dataset_default}" \
          --eval_steps "${eval_steps}" \
          --load_best_model_at_end True \
          --save_total_limit 1 \
          --include_num_input_tokens_seen True \
          --seed "${seed}" \
          --streaming True \
          --max_steps "${max_steps}" \
          >> "printout/output_file/output_${SLURM_JOB_ID}_${current_time}_${seed}.out" \
          2>> "printout/error_file/error_${SLURM_JOB_ID}_${current_time}_${seed}.err"
      done
      ;;
    *)
      echo "Unknown tulu3 variant: ${variant}" >&2
      usage
      exit 1
      ;;
  esac
}

case "${experiment}" in
  exp1)
    run_exp1 "$@"
    ;;
  exp2)
    run_exp2 "$@"
    ;;
  exp3)
    run_exp3 "$@"
    ;;
  orca)
    run_orca "$@"
    ;;
  tulu3)
    run_tulu3 "$@"
    ;;
  *)
    echo "Unknown experiment: ${experiment}" >&2
    usage
    exit 1
    ;;
esac


