#!/bin/bash
#SBATCH --partition=mbzuai
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive

module load cuda/12.4
source ~/miniconda3/bin/activate myenv
cd /mbz/users/liyuan/LLaMA-Factory
# Set NCCL to debug only errors
export NCCL_DEBUG=WARN

# Disable NVML if not required
export PYTORCH_NO_NVML=1

# Add recommended NCCL settings
export NCCL_SOCKET_IFNAME=^docker0,lo
export NCCL_IB_HCA=mlx5
export NCCL_IB_GID_INDEX=3

# Navigate to your project directory


current_time=$(date +"%Y%m%d_%H%M%S")

experiment_name="orca"
checkpoint_path="/mbz/users/liyuan/LLaMA-Factory/saves/data_mixing/${experiment_name}"
# model_base="Llama-3.2-3B"
model_base="Qwen2.5-32B"
# model_base="Llama-3.1-70B"

experiment_result="/mbz/users/liyuan/LLaMA-Factory/results/data_mixing/${experiment_name}"

domains=("t0" "cot" "flan" "niv")

lr=2.0e-6

eval_datasets=("orca_t0_val" "orca_cot_val" "orca_flan_val" "orca_niv_val")

# weight_type="original"
# dataset_name="orca_original"

# weight_type="equal"
# dataset_name="orca_equal"


weight_type="ours"
dataset_name="orca_Qwen_ours"

# weight_type="submodular"
# dataset_name="orca_submodular"

# ------------------------------------------------

seeds=(42)
for seed in "${seeds[@]}"; do
    echo "Running with seed ${seed}"
    # output_dir=""

    mkdir -p ${experiment_result}/5.2/${model_base}/${weight_type}

    for eval_dataset in "${eval_datasets[@]}"; do 
        torchrun /mbz/users/liyuan/LLaMA-Factory/scripts/cal_ppl.py \
        --model_name_or_path ${checkpoint_path}/${model_base}/${weight_type}/${model_base}_${dataset_name}_${lr}_seed42 \
        --save_name ${experiment_result}/5.2/${model_base}/${weight_type}/ppl_${eval_dataset}_${model_base}_${dataset_name}.json \
        --batch_size 1 \
        --dataset ${eval_dataset} \
        --template qwen \
        --cutoff_len 4096 \
        >> "printout/output_file/output_inference_${SLURM_JOB_ID}.out" \
            2>> "printout/error_file/error_inference_${SLURM_JOB_ID}.err"
    done
done
