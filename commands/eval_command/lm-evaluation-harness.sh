#!/bin/bash
#SBATCH --partition=mbzuai
#SBATCH --time=100:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --job-name=lm_eval

# Load necessary modules
module load cuda/12.4
source ~/miniconda3/bin/activate lm-eval

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}" || exit 1
export NCCL_DEBUG=WARN
export PYTORCH_NO_NVML=1
export HF_ALLOW_CODE_EVAL=1

mkdir -p "${PROJECT_ROOT}/printout/output_file"
mkdir -p "${PROJECT_ROOT}/printout/error_file"

current_time=$(date "+%Y.%m.%d-%H.%M.%S")

gen_kwargs="temperature=1"

output_base="${PROJECT_ROOT}/results"

# Model presets (exp_name | model_name | model_variant | relative path):
#   orca | Llama3.1-8B | original   | saves/data_mixing/orca/Llama-3.1-8B/original/Llama-3.1-8B_orca_original_1.0e-5_seed42
#   orca | Llama3.1-8B | equal      | saves/data_mixing/orca/Llama-3.1-8B/equal/Llama-3.1-8B_orca_equal_1.0e-5_seed42
#   tulu3 | Llama3.1-8B | ours      | saves/data_mixing/tulu3/Llama-3.1-8B/ours/Llama-3.1-8B_tulu3_ours_1.0e-5_seed42

exp_name="orca"
model_name="Qwen2.5-32B"
model_variant="ours"
model_path="${PROJECT_ROOT}/saves/data_mixing/orca/Qwen2.5-32B/ours/Qwen2.5-32B_orca_Qwen_ours_2.0e-6_seed42"

SLURM_JOB_ID=${SLURM_JOB_ID:-"default_job_id"}
current_time=$(date +"%Y%m%d_%H%M%S")

# Task presets:
#   ifeval | mmlu | gsm8k_cot | humaneval | toxigen | truthfulqa_mc1

# Optional task presets (comma-separated for multi-task runs):
#   mmlu_flan_cot_zeroshot | mmlu_flan_cot_fewshot | bbh_cot_fewshot
#   gpqa_main_cot_zeroshot | gsm8k_cot,gpqa_main_cot_zeroshot

tasks=("hellaswag" "agieval_nous" "toxigen" "truthfulqa_mc1")

for task in "${tasks[@]}"; do
    output_path="${output_base}/data_mixing/downstream/${exp_name}/${model_name}/${model_variant}/${task}/"
    mkdir -p "${output_path}"
    
    accelerate launch --multi_gpu --num_processes 2 \
        -m lm_eval --model hf \
        --model_args pretrained="${model_path}",parallelize=True \
        --tasks "${task}" \
        --batch_size auto \
        --seed 42 \
        --output_path "${output_path}" \
        --gen_kwargs "${gen_kwargs}" \
        --log_samples \
        --confirm_run_unsafe_code \
        --write_out \
        >> "printout/output_file/output_eval_${SLURM_JOB_ID}_${current_time}_${task}.out" \
        2>> "printout/error_file/error_eval_${SLURM_JOB_ID}_${current_time}_${task}.err"

done
