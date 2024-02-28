#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=32
#SBATCH --wait-all-nodes=1
#SBATCH --job-name=generate
#SBATCH --qos=normal
#SBATCH --account=IscrC_GELATINO
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=05:00:00
#SBATCH --output slurm_logs/data_modify-%j.out
#SBATCH --wait-all-nodes=1

export CUDA_VISIBLE_DEVICES="0,1,2,3"

source /leonardo_scratch/large/userexternal/gpuccett/data/data_venv/bin/activate

srun python -u -m synthetic_llm_data.src.abstraction_pilot.replacement_experiment \
    --data_path "/leonardo_scratch/large/userexternal/gpuccett/datasets/wemb_abstraction_data/pilot_dataset_en_utf8.csv" \
    --output_path "/leonardo_scratch/large/userexternal/gpuccett/datasets/wemb_abstraction_data/abstraction_replacement_experiment_results_postprocessed.csv" \
    --n_modifications 10 \
    --temperature 0.8 \
    --max_batch_size 32 \
    --name_or_path "../models/hf_t5/t5-11b" \
    --col_name "text"

