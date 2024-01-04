"""Complete a given text"""

import os
import re
from pathlib import Path
# from nltk import sent_tokenize
import transformers as ts
import datasets
import torch

from .utils import get_dataset_preprocessing_func
from .args import parse_complete_args
from ..utils import (save_distributed_and_collect_on_main_rank)

datasets.disable_caching()

def generate(text, llm, params, is_test=False, huggingface_or_vllm="huggingface"):
    tokenizer = (
        llm[1] if huggingface_or_vllm == "huggingface"
        else llm.get_tokenizer())

    max_new_tokens = 2048 - max(len(i) for i in tokenizer(text))

    if is_test:
        text = [i[:20] for i in text]
        max_new_tokens = 512

    if huggingface_or_vllm == "vllm":
        params.max_tokens = 2048
        generated = llm.generate(text, sampling_params=params)
        out = [out.outputs[0].text for out in generated]
        out = [re.sub(" +", " ", i + " " + j) for i,j in zip(text, out)]
        return [i[len(j):] for i,j in zip(out, text)]

    elif huggingface_or_vllm == "huggingface":
        model = llm[0]
        tokenizer.pad_token = tokenizer.eos_token
        ids = tokenizer(text,
                return_tensors='pt', padding=True, truncation=True,)

        out = model.generate(
            **{i:j.to(model.device) for i,j in ids.items()},
            max_new_tokens=max_new_tokens,
            # min_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True,
            no_repeat_ngram_size=6,
            repetition_penalty=1.05,
        )

        return [
            i[len(j):] for i,j in
            zip(tokenizer.batch_decode(out["sequences"], skip_special_tokens=True),text)
        ]

def generate_synthetic(x, llm, params, is_test, huggingface_or_vllm):
    out = {}
    out["machine_review"] = generate(
        x["prompt"], llm, params,
        is_test=is_test, huggingface_or_vllm=huggingface_or_vllm)

    out["mixed_review"] = [
        re.sub(" +", " ", i + " " + j) for i,j in
        zip(x["truncated_human_review"], out["machine_review"])]

    return out


def main(args):

    is_test = args.is_test

    huggingface_or_vllm = args.huggingface_or_vllm

    if huggingface_or_vllm == "vllm":
        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            raise ImportError('Please install vllm with "pip install vllm"')

    # local_rank = int(os.environ.get("SLURM_LOCALID", 0))
    # device = torch.device(local_rank)

    use_beam_search = True
    temp = 0.9
    topp = 0.9
    topk = 40

    global_rank = int(os.environ.get("SLURM_PROCID", 0))
    world_size = int(os.environ.get("SLURM_NTASKS", 1))

    print("Loading the data...")

    base_path = Path("." if args.base_path is None else args.base_path)
    data_files = {i:str(base_path / j) for i,j in zip(args.split_names, args.split_files)}

    data = datasets.load_dataset("csv", data_files=data_files)

    splits_uuids = {
        key : data[key]["uuid"] if "uuid" in data[key].features else None
        for key in data.keys()}

    data_keys = list(data.keys())
    data = datasets.concatenate_datasets(
        [data[key] for key in data_keys])


    if is_test and args.max_prompts is None:
        args.max_prompts = 100

    if args.max_prompts is not None:
        data = data.select(range(args.max_prompts))

    data = data.shard(num_shards=world_size, index=global_rank)
    print('Loading the model...')

    checkpoint = args.name_or_path

    if huggingface_or_vllm == "huggingface":
        model = ts.AutoModelForCausalLM.from_pretrained(
            checkpoint, torch_dtype=torch.bfloat16, device_map="auto")
        tokenizer = ts.AutoTokenizer.from_pretrained(checkpoint)
        params = None
        llm = (model, tokenizer)

    elif huggingface_or_vllm == "vllm":
        params = SamplingParams(
            temperature=temp if not use_beam_search else 0,
            top_p=topp if not use_beam_search else 1,
            top_k=topk if not use_beam_search else -1,
            max_tokens=2048,
            repetition_penalty=1.2,
            use_beam_search=use_beam_search,
            best_of=3 if use_beam_search else 1,
            # ignore_eos=True,
            skip_special_tokens=True
        )

        llm = LLM(
            model=checkpoint,
            tensor_parallel_size=args.tensor_parallel_size)

    print('Model successfully loaded.')

    preprocessing_fn = get_dataset_preprocessing_func(args)

    model_name = Path(checkpoint).name
    data = data.map(
        lambda x: preprocessing_fn(x, model_name),
        batched=True, batch_size=100,
        desc="Generating prompts")

    args.max_batch_size = 2 if is_test else args.max_batch_size
    data = data.map(lambda x:
        generate_synthetic(x, llm, params, is_test=is_test,
                           huggingface_or_vllm=huggingface_or_vllm),
        desc="Generating machine text", batched=True, batch_size=args.max_batch_size)

    final_dataset = save_distributed_and_collect_on_main_rank(
        data_shard=data, output_file=args.output_path, global_rank=global_rank,
        global_n_devices=world_size, save_after_collect=False)

    if global_rank == 0:
        for split, split_uuid in splits_uuids.items():
            if split_uuid is None:
                final_dataset.to_csv(str(args.output_path), index=False)
                break

            final_dataset.filter(lambda x: x["uuid"] in split_uuid).to_csv(
                str(Path(args.output_path) / f"{split}/{split}_{model_name}.csv"), index=False)


if __name__ == "__main__":
    main(parse_complete_args())
