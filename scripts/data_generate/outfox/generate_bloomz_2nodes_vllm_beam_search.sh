#!/bin/bash


#SBATCH --partition=boost_usr_prod
#SBATCH --time=24:00:00
#SBATCH --job-name=generate_ray_2nodes
#SBATCH --output=slurm_logs/ray_generate_2nodes-%j.log
#SBATCH --nodes=2
#SBATCH --exclusive
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=32
#SBATCH --account=IscrC_GELATINO

module purge
module load cuda
module load gcc
source /leonardo_scratch/large/userexternal/gpuccett/datasets/data_venv/bin/activate

export CUDA_VISIBLE_DEVICES="0,1,2,3"


# ===== DO NOT CHANGE THINGS HERE UNLESS YOU KNOW WHAT YOU ARE DOING =====
# This script is a modification to the implementation suggest by gregSchwartz18 here:
# https://github.com/ray-project/ray/issues/826#issuecomment-522116599
redis_password=$(uuidgen)
export redis_password


nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST") # Getting the node names
nodes_array=($nodes)

node_1=${nodes_array[0]}
# ip=$(srun --nodes=1 --ntasks=1 -w "$node_1" hostname --ip-address)

ip=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
port=6379
ip_head=$ip:$port
export ip_head
echo "IP Head: $ip_head"

echo "STARTING HEAD at $node_1"
srun --nodes=1 --ntasks=1 -w "$node_1" \
  ray start --head --node-ip-address="$ip" --port=$port --redis-password="$redis_password" --block &
sleep 10

worker_num=$(($SLURM_JOB_NUM_NODES - 1)) #number of nodes other than the head node
for ((i = 1; i <= worker_num; i++)); do
  node_i=${nodes_array[$i]}
  echo "STARTING WORKER $i at $node_i"
  echo "IP Head: $ip_head"
  this_node_ip=$(srun --nodes=1 --ntasks=1 -w "$node_i" hostname --ip-address)
  echo "Node: $node_i IP Local: $this_node_ip"
  srun --nodes=1 --ntasks=1 -w "$node_i" \
    ray start \
    --address "$ip_head" \
    --node-ip-address ${this_node_ip} \
    --num-cpus 32 \
    --num-gpus 4 \
    --redis-password="$redis_password" \
    --block &

  sleep 5
done

# ===== Call your code below =====

echo "STARTING python command"

python -m synthetic_llm_data.src.data_generation.data_generate \
    --name_or_path /leonardo_scratch/large/userexternal/gpuccett/models/hf_bloom/hf_bloomz \
    --seed 2 \
    --max_batch_size 16 \
    --use_beam_search True \
    --top_p 0.9 \
    --top_k 50 \
    --temperature 0.9 \
    --system_prompt "" \
    --max_new_tokens 412 \
    --min_new_tokens 312 \
    --dataset_name outfox \
    --prompts "/leonardo_scratch/large/userexternal/gpuccett/datasets/semeval2024-private/data/en/outfox_GPT4.jsonl" \
    --output_path /leonardo_scratch/large/userexternal/gpuccett/datasets/semeval2024-private/data/en/outfox_bloomz_beamsearch_312.jsonl \
    --human_key human_text \
    --tensor_parallel_size 8 \
    --huggingface_or_vllm vllm \
    --n_tasks 1 \
    --preprocessing "bloomz_peerread"
