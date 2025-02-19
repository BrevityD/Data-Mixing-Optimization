#!/bin/bash
#SBATCH --partition=highprio
#SBATCH --time=50:00:00
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --job-name=lm_eval
#SBATCH --output=printout/job_%j.out
#SBATCH --error=printout/job_%j.err

module load cuda/12.1
source ~/miniconda3/bin/activate myenv

cd /mbz/users/liyuan/LLaMA-Factory


# Docker image containing vLLM and dependencies
DOCKER_IMAGE="vllm/vllm-openai"

# Path to Hugging Face home on all nodes
PATH_TO_HF_HOME="/mbz/users/liyuan/.cache/huggingface"

# Additional Docker arguments if needed
ADDITIONAL_ARGS=(-e NCCL_IB_HCA=mlx5 --privileged)

# Path to the model accessible on all nodes
MODEL_PATH="/mbz/users/liyuan/LLaMA-Factory/checkpoints/Llama-3.1-70B-Instruct"

# Parallelism configuration
TENSOR_PARALLEL_SIZE=4
PIPELINE_PARALLEL_SIZE=2

# Get the head node and worker nodes
HEAD_NODE=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
WORKER_NODES=$(scontrol show hostnames $SLURM_JOB_NODELIST | tail -n +2)


# Start the head node container
srun --nodes=1 --ntasks=1 -w $HEAD_NODE \
    bash /mbz/users/liyuan/LLaMA-Factory/commands/eval_command/run_cluster.sh \
    $DOCKER_IMAGE \
    $HEAD_NODE \
    --head \
    $PATH_TO_HF_HOME \
    "${ADDITIONAL_ARGS[@]}" &

sleep 30  # Give the head node some time to start up

# Start worker nodes
for NODE in $WORKER_NODES; do
    srun --nodes=1 --ntasks=1 -w $NODE \
        bash /mbz/users/liyuan/LLaMA-Factory/commands/eval_command/run_cluster.sh \
        $DOCKER_IMAGE \
        $HEAD_NODE \
        --worker \
        $PATH_TO_HF_HOME \
        "${ADDITIONAL_ARGS[@]}" &
done

wait
echo "Ray cluster should now be running."