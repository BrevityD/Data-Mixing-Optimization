#!/bin/bash
#SBATCH --partition=mbzuai
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive

# Load necessary modules
module load cuda/12.1

# Activate your Conda environment
source ~/miniconda3/bin/activate myenv

# Set NCCL to debug only errors
export NCCL_DEBUG=WARN

# Disable NVML if not required
export PYTORCH_NO_NVML=1

# Add recommended NCCL settings
export NCCL_SOCKET_IFNAME=^docker0,lo
export NCCL_IB_HCA=mlx5
export NCCL_IB_GID_INDEX=3

# Navigate to your project directory
cd /mbz/users/liyuan/LLaMA-Factory

# Set up current time
current_time=$(date +"%Y%m%d_%H%M%S")

# Define random seeds
seeds=(42)

# Define output directory and model details
output_dir="results/bbh"
model_prefix="Meta-Llama-3-8B/full"
model_name="GPT_4o_BBH_KTO_full_KTO_12345"
dataset="logical_deduction_seven_objects"  # Define the dataset variable

# Loop over each seed
for seed in "${seeds[@]}"; do
    echo "Running evaluation with seed ${seed}"

    # Run the evaluation
    srun --ntasks=1 --gres=gpu:8 --ntasks-per-node=1 llamafactory-cli eval \
        --model_name_or_path saves/${model_prefix}/${model_name} \
        --stage sft \
        --do_predict true \
        --finetuning_type full \
        --eval_dataset ${dataset} \
        --template llama3 \
        --cutoff_len 2048 \
        --overwrite_cache true \
        --preprocessing_num_workers 32 \
        --output_dir ${output_dir}/${model_name}/${dataset} \
        --overwrite_output_dir true \
        --per_device_eval_batch_size 16 \
        --predict_with_generate true \
        --ddp_timeout 180000000 \
        --seed ${seed} \
        >> "printout/output_file/output_${current_time}_${seed}.out" \
        2>> "printout/error_file/error_${current_time}_${seed}.err"
done