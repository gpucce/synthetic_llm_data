#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --time=08:00:00
#SBATCH --job-name=generate_ray_2nodes
#SBATCH --output=slurm_logs/hf_generate_2nodes-%j.log
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=32
#SBATCH --account=IscrC_GELATINO

module purge
module load cuda
module load gcc
source /leonardo_scratch/large/userexternal/gpuccett/data/data_venv/bin/activate

export CUDA_VISIBLE_DEVICES="0,1,2,3"

srun python -m synthetic_llm_data.src.data_generation.data_generate \
    --name_or_path /leonardo_scratch/large/userexternal/gpuccett/models/hf_bloom/hf_bloomz/ \
    --seed 1 \
    --max_new_tokens 384 \
    --max_batch_size 16 \
    --max_prompts 100 \
    --use_beam_search True \
    --system_prompt "" \
    --dataset_name outfox \
    --prompts "/leonardo_scratch/large/userexternal/gpuccett/datasets/semeval2024-private/data/en/outfox_GPT4.jsonl" \
    --output_path /leonardo_scratch/large/userexternal/gpuccett/datasets/semeval2024-private/data/en/outfox_bloomz.jsonl \
    --human_key human_text \
    --tensor_parallel_size 8 \
    --huggingface_or_vllm huggingface \
    --load_in_8bit True \
    --preprocessing "bloomz_peerread"
