#!/bin/bash
#SBATCH --partition=highprio
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

experiment_name="exp2"
checkpoint_path="/mbz/users/liyuan/LLaMA-Factory/saves/data_mixing/${experiment_name}"
# model_base="Llama-3.2-3B"
model_base="Llama-3.1-8B"
# base_token="20000000"
base_token="200000000"
# base_token="5000000"

experiment_result="/mbz/users/liyuan/LLaMA-Factory/results/data_mixing/${experiment_name}"
domains=("instr" "math" "code")

sizes=(0.125 0.25 0.375 0.5 0.625 0.75)
# sizes=("one")
# Modify learning rate for different model size
lr=1.0e-5
# eval_datasets=("5000000_exp2_val")
eval_datasets=("5000000_code_val" "5000000_instr_val" "5000000_math_val" )
# eval_datasets=("code_val" "instr_val" "math_val")

sizes_1=(0.75)
sizes_2=(0.125)
for size1 in "${sizes_1[@]}"; do
    for size2 in "${sizes_2[@]}"; do
        for size3 in "${sizes[@]}"; do
            # Check if size1 + size2 + size3 == 1
            sum=$(echo "$size1 + $size2 + $size3" | bc)
            if (( $(echo "$sum == 1" | bc -l) )); then
                # Assign sizes to domains
                dataset_sizes=("$size1" "$size2" "$size3")

                # Construct dataset names
                dataset_instr="${base_token}_${domains[0]}_${dataset_sizes[0]}"
                dataset_math="${base_token}_${domains[1]}_${dataset_sizes[1]}"
                dataset_code="${base_token}_${domains[2]}_${dataset_sizes[2]}"


                mkdir -p ${experiment_result}/${model_base}/${base_token}

                for eval_dataset in "${eval_datasets[@]}"; do 
                    torchrun /mbz/users/liyuan/LLaMA-Factory/scripts/cal_ppl.py \
                    --model_name_or_path ${checkpoint_path}/${model_base}/${base_token}/${model_base}_${dataset_instr}_${dataset_math}_${dataset_code}_${lr}_seed42 \
                    --save_name ${experiment_result}/${model_base}/${base_token}/ppl_${eval_dataset}_${model_base}_${dataset_instr}_${dataset_math}_${dataset_code}.json \
                    --batch_size 2 \
                    --dataset ${eval_dataset} \
                    --template llama3 \
                    --cutoff_len 4096 \
                    >> "printout/output_file/output_inference_${SLURM_JOB_ID}.out" \
                        2>> "printout/error_file/error_inference_${SLURM_JOB_ID}.err"
                done
            fi
        done
    done  
done