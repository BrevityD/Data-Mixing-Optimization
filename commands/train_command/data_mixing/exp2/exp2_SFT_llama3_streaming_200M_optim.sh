#!/bin/bash
#SBATCH --partition=highprio
#SBATCH --time=500:00:00
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive


# Load necessary modules

module load cuda/12.4
source ~/miniconda3/bin/activate myenv
cd /mbz/users/liyuan/LLaMA-Factory

nvidia-smi
export PYTORCH_NO_NVML=1
export NCCL_SOCKET_IFNAME=^docker0,lo
export NCCL_IB_HCA=mlx5
export NCCL_IB_GID_INDEX=3
export TORCH_USE_CUDA_DSA=1

# Navigate to your project directory


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

model_name="Llama-3.1-8B"
# model_name="Llama-3.2-3B"
base_token=200000000
domains=("instr" "math" "code")

lr=1.0e-5

# Construct dataset names
dataset_instr="${base_token}_${domains[0]}_${base_token}_${model_name}_${domains[0]}_optim"
dataset_math="${base_token}_${domains[1]}_${base_token}_${model_name}_${domains[1]}_optim"
dataset_code="${base_token}_${domains[2]}_${base_token}_${model_name}_${domains[2]}_optim"

for seed in "${seeds[@]}"; do
    echo "Running with seed ${seed}"
    output_dir="saves/data_mixing/exp2/${model_name}/${base_token}/optim/${model_name}_${dataset_instr}_${dataset_math}_${dataset_code}_${lr}_seed${seed}"

    srun torchrun \
        --nproc_per_node=8 \
        --nnodes=8 \
        --rdzv_id=$SLURM_JOB_ID \
        --rdzv_backend=c10d \
        --rdzv_endpoint=${head_node_ip}:29500 \
        src/train.py \
        --model_name_or_path checkpoints/${model_name} \
        --stage sft \
        --do_train \
        --finetuning_type full \
        --deepspeed "examples/deepspeed/ds_z3_config.json" \
        --dataset ${dataset_instr},${dataset_math},${dataset_code} \
        --template llama3 \
        --cutoff_len 4096 \
        --overwrite_cache \
        --output_dir ${output_dir} \
        --logging_steps 50 \
        --save_steps 50 \
        --plot_loss \
        --overwrite_output_dir \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 4 \
        --learning_rate ${lr} \
        --dispatch_batches False \
        --lr_scheduler_type cosine \
        --warmup_ratio 0.05 \
        --bf16 \
        --ddp_timeout 180000 \
        --per_device_eval_batch_size 8 \
        --eval_strategy steps \
        --eval_dataset 5000000_exp2_val \
        --eval_steps 50 \
        --load_best_model_at_end True \
        --save_total_limit 1 \
        --include_num_input_tokens_seen True \
        --seed "${seed}" \
        --streaming True \
        --max_steps 2000 \
        >> "printout/output_file/output_${SLURM_JOB_ID}_${current_time}_${seed}.out" \
        2>> "printout/error_file/error_${SLURM_JOB_ID}_${current_time}_${seed}.err"
done
