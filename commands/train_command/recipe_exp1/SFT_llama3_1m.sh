#!/bin/bash
#SBATCH --partition=highprio
#SBATCH --time=500:00:00
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --exclude=g42-h100-instance-075,g42-h100-instance-078,g42-h100-instance-079,g42-h100-instance-089

# Load necessary modules
module load cuda/12.1

# Activate your Conda environment
source ~/miniconda3/bin/activate myenv

# Disable NVML if not required
export PYTORCH_NO_NVML=1

# Add recommended NCCL settings
export NCCL_SOCKET_IFNAME=^docker0,lo
export NCCL_IB_HCA=mlx5
export NCCL_IB_GID_INDEX=3
# export NCCL_P2P_LEVEL=NVL

# Navigate to your project directory
cd /mbz/users/liyuan/LLaMA-Factory
mkdir -p printout/output_file
mkdir -p printout/error_file

# Set up current time
current_time=$(date +"%Y%m%d_%H%M%S")

# Define head_node_ip using SLURM's environment variables
head_node=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
head_node_ip=$(getent hosts "$head_node" | awk '{ print $1 }')

# Verify head_node_ip
echo "Head node: $head_node ($head_node_ip)"

# Set seeds
seeds=(42)  # Add more seeds if needed

# model_name="Llama-3.1-8B"
model_name="Llama-3.2-3B"

dataset1_base="openmathinstruct2_5M"
dataset2_base="opencoder-sft"

# Define arrays of total_num and prop1 values
total_nums=(1000000)
prop1_values=(0.125 0.25 0.375 0.5 0.625 0.75 0.875)
# prop1_values=(0.25 0.5 0.625)
lr=2.0e-5

# Loop over total_num and prop1 values
for total_num in "${total_nums[@]}"; do
    for prop1 in "${prop1_values[@]}"; do
        # Calculate num_dataset1 and num_dataset2
        num_dataset1=$(python -c "print(int(${total_num}*${prop1}))")
        num_dataset2=$(python -c "print(int(${total_num}-${num_dataset1}))")

        dataset_1="${dataset1_base}_${num_dataset1}"
        dataset_2="${dataset2_base}_${num_dataset2}"

        # Calculate max_steps, logging_steps, save_steps, eval_steps
        max_steps=$(python -c "import math; print(math.ceil(${total_num}*3/512))")

        # logging_steps, save_steps, eval_steps = floor(total_num/10)
        interval=$(python -c "import math; print(int(${max_steps}//10))")

        # Loop over each seed
        for seed in "${seeds[@]}"; do
            echo "Running with seed ${seed}"
            echo "total_num: ${total_num}, prop1: ${prop1}, num_dataset1: ${num_dataset1}, num_dataset2: ${num_dataset2}"
            echo "max_steps: ${max_steps}, interval: ${interval}"

            output_dir="saves/recipe_exp1/${model_name}/${total_nums}/${model_name}_${dataset_1}_${dataset_2}_${lr}_seed${seed}"

            srun torchrun \
                --nproc_per_node=8 \
                --nnodes=16 \
                --rdzv_id=$SLURM_JOB_ID \
                --rdzv_backend=c10d \
                --rdzv_endpoint=${head_node_ip}:29500 \
                src/train.py \
                --model_name_or_path checkpoints/${model_name} \
                --stage sft \
                --do_train \
                --finetuning_type full \
                --deepspeed "examples/deepspeed/ds_z3_config.json" \
                --dataset ${dataset_1},${dataset_2} \
                --template llama3 \
                --cutoff_len 4096 \
                --overwrite_cache \
                --streaming True \
                --output_dir ${output_dir} \
                --logging_steps ${interval} \
                --save_steps ${interval} \
                --eval_steps ${interval} \
                --plot_loss \
                --overwrite_output_dir \
                --per_device_train_batch_size 2 \
                --gradient_accumulation_steps 2 \
                --learning_rate ${lr} \
                --max_steps  ${max_steps} \
                --dispatch_batches False \
                --lr_scheduler_type linear \
                --warmup_ratio 0.03 \
                --bf16 \
                --ddp_timeout 180000000 \
                --per_device_eval_batch_size 4 \
                --eval_strategy steps \
                --eval_dataset recipe_exp1_val \
                --load_best_model_at_end True \
                --save_total_limit 1 \
                --seed "${seed}" \
                >> "printout/output_file/output_${SLURM_JOB_ID}_${current_time}_${seed}.out" \
                2>> "printout/error_file/error_${SLURM_JOB_ID}_${current_time}_${seed}.err"
        done
    done
done