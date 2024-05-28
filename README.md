# Synthetic LLM data

Code for the paper __"AI 'News' Content Farms are easy to make and hard to detect: a case study in Italian"__


## Model Fine Tuning
Our fine-tunings are all done through [llm-foundry](https://github.com/mosaicml/llm-foundry/) on the [change-it](https://huggingface.co/datasets/gsarti/change_it) dataset.

In the folder [foundry_yamls](foundry_yamls) you can find the yamls used to fine-tune the models, and we refer to the llm-foundry documentation for how to set up the fine-tuning. In particular, the [change_it](foundry_yamls/change_it/) folder contains all the files used to fine-tune on Italian.

## Synthetic Text Detection
The [experiments](experiments) folder contains the sbatch files to run all the experiments in the paper, after the fine-tunings have been run.
 - To generate the synthetic datasets, use the scripts in [ita_data_generate](experiments/ita_data_generate/). 
 - To run the detection experiments with proxy models the scripts to use are in  [proxy_models_comparisons](experiments/proxy_model_comparisons/).
 - The supervised detection experiments can be found in [supervised_detection](experiments/supervised_detection/)
 - For the experiments replicas on the [xsum](https://huggingface.co/datasets/EdinburghNLP/xsum) the fine-tuning yamls  are available in [xsum](foundry_yamls/xsum) and the experiments with fine-tuned models are available in [xsum_experiments](experiments/xsum_experiments/) with a similar structure as those for Italian. 


An example experiment, after fine-tuning llama on the change-it dataset to generate the synthetic texts one can run 
```
sbatch experiments/ita_data_generate/ita_dat_generate_hf_llama/llama-7b_change_it.sbatch
```
possibly adjusting the experiment file `experiments/ita_data_generate/ita_dat_generate_hf_llama/llama-7b_change_it.sbatch` to the fine-tuned model and to the dataset by adjusting the values for 

```
--name-or-path
--modifier-model
--model-name
--data-path
```

where `--name-or-path` is the path to the model that generates the synthetic texts, `--modifier-model` is the path to the model used to create text variations as in [DetectGPT](https://github.com/eric-mitchell/detect-gpt) (we always use [it5](https://huggingface.co/gsarti/it5-base) for italian) and `--model-path` is the path to the model that computes the likelihood.

Instead `--data-path` is the path to the dataset.


