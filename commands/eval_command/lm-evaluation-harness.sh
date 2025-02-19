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
cd /mbz/users/liyuan/LLaMA-Factory

# Set NCCL to debug only errors
export NCCL_DEBUG=WARN

# Disable NVML if not required
export PYTORCH_NO_NVML=1
export HF_ALLOW_CODE_EVAL=1

mkdir -p /mbz/users/liyuan/LLaMA-Factory/printout/output_file
mkdir -p /mbz/users/liyuan/LLaMA-Factory/printout/error_file

# Run the lm_eval command
current_time=$(date "+%Y.%m.%d-%H.%M.%S")

# Define task and generation arguments
# task="mmlu_flan_cot_zeroshot"
# task="mmlu_flan_cot_fewshot"
# task="bbh_cot_fewshot"
# task="gpqa_main_cot_zeroshot"

# task="gsm8k_cot,gpqa_main_cot_zeroshot"
gen_kwargs="temperature=1"

output_base="/mbz/users/liyuan/LLaMA-Factory/results"

# model_path="/mbz/users/liyuan/LLaMA-Factory/saves/data_mixing/orca/Llama-3.1-8B/original/Llama-3.1-8B_orca_original_1.0e-5_seed42"
# model_name="Llama3.1-8B"
# model_variant="original"

# model_path="/mbz/users/liyuan/LLaMA-Factory/saves/data_mixing/orca/Llama-3.1-8B/equal/Llama-3.1-8B_orca_equal_1.0e-5_seed42"
# model_name="Llama3.1-8B"
# model_variant="equal"

# model_path="/mbz/users/liyuan/LLaMA-Factory/saves/data_mixing/orca/Llama-3.1-70B/original/Llama-3.1-70B_orca_original_1.0e-6_seed42"
# model_name="Llama3.1-70B"
# model_variant="original"

# model_path="/mbz/users/liyuan/LLaMA-Factory/saves/data_mixing/orca/Llama-3.1-70B/equal/Llama-3.1-70B_orca_equal_1.0e-6_seed42"
# model_name="Llama3.1-70B"
# model_variant="equal"

# model_path="/mbz/users/liyuan/LLaMA-Factory/saves/data_mixing/orca/Qwen2.5-32B/equal/Qwen2.5-32B_orca_equal_2.0e-6_seed42"
# model_name="Qwen2.5-32B"
# model_variant="equal"

# model_path="/mbz/users/liyuan/LLaMA-Factory/saves/data_mixing/orca/Qwen2.5-32B/original/Qwen2.5-32B_orca_original_2.0e-6_seed42"
# model_name="Qwen2.5-32B"
# model_variant="our"

# model_path="/mbz/users/liyuan/LLaMA-Factory/saves/data_mixing/orca/Qwen2.5-32B/submodular/Qwen2.5-32B_orca_submodular_2.0e-6_seed42"
# model_name="Qwen2.5-32B"
# model_variant="submodular"

# model_path="/mbz/users/liyuan/LLaMA-Factory/saves/data_mixing/orca/Llama-3.1-8B/submodular/Llama-3.1-8B_orca_submodular_1.0e-5_seed42"
# model_name="Llama3.1-8B"
# model_variant="submodular"

# exp_name="tulu3"
# model_path="/mbz/users/liyuan/LLaMA-Factory/saves/data_mixing/tulu3/Llama-3.1-8B/submodular/Llama-3.1-8B_tulu3_submodular_1.0e-5_seed42"
# model_name="Llama3.1-8B"
# model_variant="submodular"


# model_path="/mbz/users/liyuan/LLaMA-Factory/saves/data_mixing/tulu3/Llama-3.1-8B/ours/Llama-3.1-8B_tulu3_ours_1.0e-5_seed42"
# model_path="/mbz/users/liyuan/LLaMA-Factory/saves/data_mixing/tulu3/Llama-3.1-8B/ours/Llama-3.1-8B_tulu3_ours_1.0e-5_seed42"
# model_name="Llama3.1-8B"
# model_variant="ours"




# exp_name="orca"
# model_path="/mbz/users/liyuan/LLaMA-Factory/saves/data_mixing/orca/Llama-3.1-8B/ours/Llama-3.1-8B_orca_ours_1.0e-5_seed42"
# model_name="Llama3.1-8B"
# model_variant="ours"


exp_name="orca"
model_path="/mbz/users/liyuan/LLaMA-Factory/saves/data_mixing/orca/Qwen2.5-32B/ours/Qwen2.5-32B_orca_Qwen_ours_2.0e-6_seed42"
# model_name="Llama3.1-70B"
model_name="Qwen2.5-32B"
model_variant="ours"

SLURM_JOB_ID=${SLURM_JOB_ID:-"default_job_id"}
current_time=$(date +"%Y%m%d_%H%M%S")

# Loop through tasks and run the command for each task




# tasks=("ifeval")

# tasks=("mmlu" "gsm8k_cot" "humaneval" "toxigen" "truthfulqa_mc1")

tasks=("hellaswag" "agieval_nous" "toxigen" "truthfulqa_mc1")

# tasks=("agieval_nous")

for task in "${tasks[@]}"; do
    output_path="${output_base}/data_mixing/downstream/${exp_name}/${model_name}/${model_variant}/${task}/"
    
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
