python -u -m synthetic_llm_data.src.synthetic_detection.detect_gpt.dataset_lm_modification \
    --output-path "small_test" \
    --modifier-model "/leonardo_scratch/large/userexternal/gpuccett/models/hf_t5/t5-small" \
    --model-name "/leonardo_scratch/large/userexternal/gpuccett/models/hf_gpt/gpt2-hf" \
    --name-or-path "/leonardo_scratch/large/userexternal/gpuccett/models/hf_gpt/gpt2-hf" \
    --data-path "/leonardo_scratch/large/userexternal/gpuccett/data/xsum" \
    --do-generation \
    --col-name "document" \
    --do-modification \
    --n-modifications 2 \
    --human-key "document" \
    --max-batch-size 16 \
    --dtype float32 \
    --dataset-type disk \
    --padding-side left \
    --min-new-tokens 150 \
    --max-new-tokens 200 \
    --max-seq-len 150 \
    --preprocessing xsum \
    --n-samples 100 \
    --seed 10

