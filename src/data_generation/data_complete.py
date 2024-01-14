"""Complete a given text"""

import os
import re
from pathlib import Path
import datasets
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

from .utils import get_min_length_logits_processor
from .prompts import PromptPreprocessor
from .args import parse_complete_args
from ..utils import save_distributed_and_collect_on_main_rank

datasets.disable_caching()


def get_model_and_tokenizer(args):

    if args.huggingface_or_vllm == "huggingface":
        llm = AutoModelForCausalLM.from_pretrained(
            args.name_or_path, torch_dtype=torch.bfloat16, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(args.name_or_path)

    elif args.huggingface_or_vllm == "vllm":

        llm = LLM(
            model=args.name_or_path,
            tensor_parallel_size=args.tensor_parallel_size)

        tokenizer = llm.get_tokenizer()

    return llm, tokenizer

def generate(text, llm, tokenizer, args):

    max_new_tokens = 2048 - max(len(i) for i in tokenizer(text))

    if args.huggingface_or_vllm == "vllm":
        min_length_logits_processor = get_min_length_logits_processor(
            args.min_new_tokens, tokenizer.eos_token_id)

        params = SamplingParams(
            temperature=args.temperature if not args.use_beam_search else 0,
            top_p=args.top_p if not args.use_beam_search else 1,
            top_k=args.top_k if not args.use_beam_search else -1,
            max_tokens=2048,
            repetition_penalty=1.05,
            use_beam_search=args.use_beam_search,
            best_of=3 if args.use_beam_search else 1,
            skip_special_tokens=True,
            logits_processors=[min_length_logits_processor]
        )
        generated = llm.generate(text, sampling_params=params)
        out = [out.outputs[0].text for out in generated]
        out = [re.sub(" +", " ", prompt + " " + generation)
               for prompt, generation in zip(text, out)]
        return [generation[len(prompt):] for prompt, generation in zip(text, out)]

    elif args.huggingface_or_vllm == "huggingface":
        tokenizer.pad_token = tokenizer.eos_token
        ids = tokenizer(
            text, return_tensors='pt', padding=True, truncation=True, max_length=256)

        out = llm.generate(
            **{i:j.to(llm.device) for i,j in ids.items()},
            max_new_tokens=max_new_tokens,
            min_new_tokens=args.min_new_tokens,
            top_p=args.top_p,
            do_sample=True,
            # top_k=args.top_k,
            # temperature=args.temperature,
            return_dict_in_generate=True,
            # no_repeat_ngram_size=6,
            # repetition_penalty=1.05,
        )

        return [
            generation[len(prompt):] for generation, prompt in
            zip(tokenizer.batch_decode(out["sequences"], skip_special_tokens=True), text)
        ]

def generate_synthetic(x, llm, tokenizer, args):
    out = {}
    out["machine_text"] = generate(x["prompt"], llm, tokenizer, args)
    out["mixed_text"] = [
        re.sub(" +", " ", i + " " + j) for i,j in
        zip(x["truncated_human_text"], out["machine_text"])]

    return out

def generation(
        args, data, clean_temp_files=True, output_on_all_nodes=False):

    global_rank = int(os.environ.get("SLURM_PROCID", 0))
    world_size = int(os.environ.get("SLURM_NTASKS", 1))
    data = data.shard(num_shards=world_size, index=global_rank)

    preprocessing_fn = PromptPreprocessor(args)
    data = data.map(preprocessing_fn, desc="Generating prompts")

    print('Loading the model...')
    llm, tokenizer = get_model_and_tokenizer(args)
    print('Model successfully loaded.')

    data = data.map(lambda x:
        generate_synthetic(x, llm, tokenizer, args),
        desc="Generating machine text", batched=True, batch_size=args.max_batch_size)

    final_dataset = save_distributed_and_collect_on_main_rank(
        data_shard=data, output_file=args.output_path, global_rank=global_rank,
        global_n_devices=world_size, save_after_collect=False,
        clean_temp_files=clean_temp_files, output_on_all_nodes=output_on_all_nodes)

    return final_dataset

def main(args):

    global_rank = int(os.environ.get("SLURM_PROCID", 0))
    world_size = int(os.environ.get("SLURM_NTASKS", 1))

    print("Loading the data...")

    base_path = Path(args.base_path)
    data_files = {i:str(base_path / j) for i, j in zip(args.split_names, args.split_files)}
    data = datasets.load_dataset("csv", data_files=data_files)

    splits_uuids = {
        key : data[key]["uuid"] if "uuid" in data[key].features else None
        for key in data.keys()}

    data_keys = list(data.keys())
    data = datasets.concatenate_datasets(
        [data[key] for key in data_keys])

    if args.max_prompts is not None:
        data = data.select(range(args.max_prompts))

    final_dataset = generation(args, data)

    if global_rank == 0:
        if args.columns_to_remove is not None:
            final_dataset = final_dataset.remove_columns(args.columns_to_remove)
        for split, split_uuid in splits_uuids.items():
            if split_uuid is None:
                final_dataset.to_csv(str(args.output_path), index=False)
                break

            model_name = Path(args.name_or_path).name
            final_dataset.filter(lambda x: x["uuid"] in split_uuid).to_csv(
                str(Path(args.output_path) / f"{split}/{split}_{model_name}.csv"), index=False)

if __name__ == "__main__":
    main(parse_complete_args())
