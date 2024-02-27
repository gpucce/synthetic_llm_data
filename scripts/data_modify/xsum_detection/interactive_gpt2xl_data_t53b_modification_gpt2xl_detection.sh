#!/bin/bash

python -u -m synthetic_llm_data.src.synthetic_detection.detect_gpt.dataset_lm_modification \
    --output-path "detectGPT_experiments/xsum/gpt2xlgen_t5-3bmodif_gpt2xldetect" \
    --modifier-model "/leonardo_scratch/large/userexternal/gpuccett/models/hf_t5/t5-3b" \
    --model-name "/leonardo_scratch/large/userexternal/gpuccett/models/hf_gpt/gpt2-xl-hf" \
    --data-path "/leonardo_scratch/large/userexternal/gpuccett/datasets/xsum" \
    --name-or-path "/leonardo_scratch/large/userexternal/gpuccett/models/hf_gpt/gpt2-xl-hf" \
    --huggingface-or-vllm "huggingface" \
    --col-name "document" \
    --human-key "document" \
    --max-batch-size 16 \
    --do-generation \
    --dtype "float32" \
    --temperature 1.0 \
    --n-modifications 3 \
    --dataset-type disk \
    --min-new-tokens 200 \
    --max-new-tokens 200 \
    --max-seq-len 150 \
    --preprocessing xsum \
    --padding-side "left" \
    --n-samples 300 \
    --seed 10