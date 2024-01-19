#!/bin/bash

python -u -m synthetic_llm_data.src.abstraction_pilot.abstraction_replacement_experiment \
    --data-path "/leonardo_scratch/large/userexternal/gpuccett/data/wemb_abstraction_data/pilot_dataset_en_utf8.csv" \
    --output-path "/leonardo_scratch/large/userexternal/gpuccett/data/wemb_abstraction_data/abstraction_replacement_experiment_results_postprocessed.csv" \
    --n-modifications 2 \
    --temperature 0.8 \
    --max-batch-size 32 \
    --model-name-or-path "../models/hf_t5/t5-small" \
    --col-name "text"

