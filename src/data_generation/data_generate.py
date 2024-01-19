
import os
import json
from pathlib import Path

import torch
import datasets

from .args import parse_generate_args
from .utils import (
    get_preprocessing_func,
    save_to_right_format,
    get_min_length_logits_processor
)
from ..utils import save_distributed_and_collect_on_main_rank


datasets.disable_caching()

def get_model_tokenizer_and_params(args):
    if args.huggingface_or_vllm == "vllm":
        from vllm import LLM, SamplingParams
        llm = LLM(model=args.name_or_path, seed=args.seed,
            tensor_parallel_size=args.tensor_parallel_size)
        tokenizer = llm.get_tokenizer()

        min_length_logits_processor = get_min_length_logits_processor(
            args.min_new_tokens, tokenizer.eos_token_id)

        params = SamplingParams(
            temperature=args.temperature if not args.use_beam_search else 0,
            top_p=args.top_p if not args.use_beam_search else 1,
            top_k=args.top_k if not args.use_beam_search else -1,
            max_tokens=args.max_new_tokens,
            repetition_penalty=1.2,
            use_beam_search=args.use_beam_search,
            best_of=3 if args.use_beam_search else 1,
            # ignore_eos=True,
            skip_special_tokens=True,
            logits_processors=[min_length_logits_processor]
        )
    elif args.huggingface_or_vllm == "huggingface":
        from transformers import AutoModelForCausalLM, AutoTokenizer
        params = None
        llm = AutoModelForCausalLM.from_pretrained(
            args.name_or_path, torch_dtype=torch.bfloat16,
            device_map="auto", load_in_8bit=args.load_in_8bit)
        tokenizer = AutoTokenizer.from_pretrained(args.name_or_path)

    return llm, tokenizer, params

def generate(text, llm, tokenizer, args, params, is_test=False, huggingface_or_vllm="huggingface"):

    max_tokens = args.max_new_tokens
    if is_test:
        text = [i[:20] for i in text]
    if huggingface_or_vllm == "vllm":
        params.max_tokens = max_tokens
        generated = llm.generate(text, sampling_params=params)
        return [out.outputs[0].text for out in generated]

    elif huggingface_or_vllm == "huggingface":
        tokenizer.pad_token = tokenizer.eos_token
        ids = tokenizer(text, return_tensors='pt', padding=True, truncation=True)

        out = llm.generate(
            **{i:j.to(llm.device) for i,j in ids.items()},
            max_new_tokens=max_tokens,
            min_new_tokens=args.min_new_tokens,
            return_dict_in_generate=True,
            output_scores=True,
            no_repeat_ngram_size=6,
            repetition_penalty=1.05,
        )

        full_out = tokenizer.batch_decode(
            out["sequences"], skip_special_tokens=True)

        return [gen[len(prompt):] for prompt, gen in zip(text, full_out)]

def generate_synthetic(
    prompt_dict, llm, tokenizer, params, args):

    is_test = args.is_test

    preprocessing_fn = get_preprocessing_func(args)

    new_prompt_dict = {}
    new_prompt_dict["prompt"] = preprocessing_fn(prompt_dict)
    new_prompt_dict["human_text"] = prompt_dict[args.human_key]
    new_prompt_dict["machine_text"] = generate(
        new_prompt_dict["prompt"], llm, tokenizer, args, params,
        is_test=is_test, huggingface_or_vllm=args.huggingface_or_vllm)

    new_prompt_dict["model"] = [(
        args.name_or_path.split("/")[-1]
        if len(args.name_or_path.split("/")[-1]) > 0
        else args.name_or_path.split("/")[-2]
    ) for _ in prompt_dict[args.human_key]]

    new_prompt_dict["source"] = [
        args.dataset_name for _ in prompt_dict[args.human_key]]
    new_prompt_dict["source_ID"] = prompt_dict.get(
        "source_ID", ["" for _ in prompt_dict[args.human_key]])

    for key in prompt_dict:
        if key not in new_prompt_dict:
            new_prompt_dict[key] = prompt_dict[key]

    return new_prompt_dict

def generation(args, data):
    llm, tokenizer, params = get_model_tokenizer_and_params(args)

    data = data.map(
        lambda x: generate_synthetic(
            x, llm, tokenizer, params, args), desc="Generating machine text",
        batched=True, batch_size=args.max_batch_size)

    return data

def main(args):

    ntasks = int(os.environ.get("SLURM_NTASKS", 1)) if args.n_tasks is None else args.n_tasks
    args.ntasks = ntasks

    procid = int(os.environ.get("SLURM_PROCID", 0))
    args.procid = procid

    print(f"########################### Tensor Parallel {args.tensor_parallel_size}")
    print(f"########################### ntasks {ntasks}")
    print(f"########################### procid {procid}")

    if args.is_test:
        args.name_or_path = "/leonardo_scratch/large/userexternal/" \
            "gpuccett/models/hf_gpt/gpt2-hf"

    dataset_type = "csv" if args.prompts.endswith(".csv") else "json"
    prompt_dicts = datasets.load_dataset(dataset_type, data_files=args.prompts)["train"]
    if "prompts" in prompt_dicts.column_names:
        prompt_dicts = prompt_dicts.rename_column("prompts", "prompt")

    if args.max_prompts is not None:
        prompt_dicts = prompt_dicts.select(range(args.max_prompts))

    if args.system_prompt is None:
        args.system_prompt = "[INST]<<SYS>>\n \n<</SYS>>\n\n {} [/INST]"

    if len(args.system_prompt) > 0:
        prompt_dicts = prompt_dicts.map(
            lambda x: {"prompt": args.system_prompt.format(x["prompt"])})

    prompt_dicts = prompt_dicts.shard(num_shards=ntasks, index=procid)

    future_df_path = Path(args.output_path)

    temp_file_path = f"{future_df_path}_temp_procid_{procid}.jsonl"
    done_df = datasets.Dataset.from_dict({})
    if os.path.exists(temp_file_path):
        if os.stat(temp_file_path).st_size > 0:
            done_df = datasets.load_dataset("json", data_files=temp_file_path)["train"]
        os.remove(temp_file_path)

    prompt_dicts = generation(args, prompt_dicts)

    if args.columns_to_remove is not None:
        prompt_dicts = prompt_dicts.remove_columns(args.columns_to_remove)

    prompt_dicts = save_distributed_and_collect_on_main_rank(
        data_shard=prompt_dicts, global_rank=procid, global_n_devices=ntasks,
        save_after_collect=False, output_file=str(future_df_path).replace(".jsonl", "")
            .replace(".json", "").replace(".csv", ""))

    if procid == 0:
        save_to_right_format(prompt_dicts, future_df_path)

if __name__ == "__main__":
    main(parse_generate_args())
