# Synthetic LLM data

Code for the paper __"AI 'News' Content Farms are easy to make and hard to detect: a case study in Italian"__


## Model Fine Tuning
Our fine-tunings are all done through [llm-foundry](https://github.com/mosaicml/llm-foundry/) on the [change-it](https://huggingface.co/datasets/gsarti/change_it) dataset.

In the folder [foundry_yamls](foundry_yamls) you can find the yamls used to fine-tune the models, and we refer to the llm-foundry documentation for how to set up the fine-tuning.

## Synthetic Text Detection
The [experiments](experiments) folder contains the sbatch files to run all the experiments in the paper, after the fine-tunings have been run.
 - To generate the synthetic datasets, use the scripts in [ita_data_generate](experiments/ita_data_generate/). 
 - To run the detection experiments with proxy models the scripts to use are in  [proxy_models_comparisons](experiments/proxy_model_comparisons/).
 - The supervised detection experiments can be found in [supervised_detection](experiments/supervised_detection/)