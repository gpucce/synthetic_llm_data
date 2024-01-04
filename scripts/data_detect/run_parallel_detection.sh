N=4
(
for j in {1..4}
do
    ((i=i%N)); ((i++==0)) && wait
    export CUDA_VISIBLE_DEVICES=$i
    python -m data_handling.synthetic_detection \
        --data_path /leonardo_work/IscrC_GELATINO/gpuccett/Repos/semeval2024-private/data/other_languages/extra_italian_news_camoscio.jsonl \
        --model_name_or_path ../models/hf_roberta/xlm-roberta-base/ \
        --seed $i &
done
)