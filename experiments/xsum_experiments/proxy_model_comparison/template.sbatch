#!/bin/bash

#SBATCH --nodes={nnodes}
#SBATCH --ntasks-per-node={taskspernode}
#SBATCH --cpus-per-task={cpuspertask}
#SBATCH --gpus-per-task={gpuspertask}
#SBATCH --wait-all-nodes=1
#SBATCH --job-name=generate
#SBATCH --account=IscrC_GELATINO
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=24:00:00
#SBATCH --output {logs}

export CUDA_VISIBLE_DEVICES="0,1,2,3"

cd /leonardo_scratch/large/userexternal/gpuccett/data
source /leonardo_scratch/large/userexternal/gpuccett/data/data_venv/bin/activate

srun python -m synthetic_llm_data.src.synthetic_detection.detect_gpt.dataset_lm_modification \
    --model-name /leonardo_scratch/large/userexternal/gpuccett/models/{lossmodel} \
    --data-path {datapath} \
    --output-path {outputpath} \
    --seed 1 \
    --human-key document \
    --col-name document \
    --max-batch-size {maxbatchsize} \
    --huggingface-or-vllm huggingface \
    --dataset-type disk \
    --length-filter 0 \
    --dtype float16 \
    --n-modifications 20 \
    --temperature 1.0 \
    --do-compute-loss
