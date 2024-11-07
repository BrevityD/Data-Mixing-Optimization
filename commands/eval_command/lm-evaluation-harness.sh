#!/bin/bash
#SBATCH --partition=mbzuai
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --job-name=lm_eval

# Load necessary modules
module load cuda/12.1

# Activate your Conda environment
source ~/miniconda3/bin/activate myenv

# Set NCCL to debug only errors
export NCCL_DEBUG=WARN

# Disable NVML if not required
export PYTORCH_NO_NVML=1

cd /mbz/users/liyuan/LLaMA-Factory

mkdir -p /mbz/users/liyuan/LLaMA-Factory/printout/output_file
mkdir -p /mbz/users/liyuan/LLaMA-Factory/printout/error_file

# Run the lm_eval command
current_time=$(date "+%Y.%m.%d-%H.%M.%S")

# Define task and generation arguments
task="gsm8k"
gen_kwargs="temperature=1,top_p=1"
model_path="/mbz/users/liyuan/LLaMA-Factory/checkpoints/Meta-Llama-3.1-8B-Instruct"
model_name=$(python -c "print('${model_path}'.split('/')[-1])")
# Define output_path with the extracted component
output_path="/mbz/users/liyuan/LLaMA-Factory/results/${task}/${model_name}"


# --num_fewshot 0 \


# Run the lm_eval command and redirect output and error logs
lm_eval --model hf \
    --model_args pretrained=${model_path} \
    --tasks "${task}" \
    --device cuda \
    --gen_kwargs "${gen_kwargs}" \
    --seed 42 \
    --batch_size 32 \
    --output_path ${output_path} \
    --log_samples \
    --write_out \
    >> "printout/output_file/output_eval_${current_time}.out" \
    2>> "printout/error_file/error_eval_${current_time}.err"