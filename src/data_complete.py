"""Complete a given text"""

import os
import re
from argparse import ArgumentParser
from pathlib import Path
# from nltk import sent_tokenize
import transformers as ts
import datasets
import torch

from .preprocessing import get_semeval_task3_prompt
from .utils import (str2bool, save_distributed_and_collect_on_main_rank)

datasets.disable_caching()

def generate(text, llm, params, is_test=False, huggingface_or_vllm="huggingface"):
    tokenizer = (
        llm[1] if huggingface_or_vllm == "huggingface"
        else llm.get_tokenizer())

    max_new_tokens = 2048 - max([len(i) for i in tokenizer(text)])

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
    out = x
    out["machine_review"] = generate(
        out["prompt"], llm, params,
        is_test=is_test, huggingface_or_vllm=huggingface_or_vllm)

    out["mixed_review"] = [i+j for i,j in zip(out["truncated_human_review"], out["machine_review"])]
    return out

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--is_test", type=str2bool, default=True)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument(
        "--huggingface_or_vllm",
        type=str,
        default="huggingface",
        choices=["huggingface", "vllm"])
    return parser.parse_args()

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

    # HF dataset
    # data = read_PeerRead("/leonardo_scratch/large/userexternal/gpuccett/data/PeerRead/")
    # data_files = "/leonardo_scratch/large/userexternal/gpuccett/data/PeerRead_full.jsonl"
    # data = datasets.load_dataset(
    #     "json", data_files=data_files)

    base_path = Path("/leonardo_scratch/large/userexternal/gpuccett/data/")
    base_path /= "semeval2024-private/semeval-taskC/data/"
    data_files = {
        "train": str(base_path / "train/train_chatgpt.csv"),
        "dev":   str(base_path / "dev/dev_chatgpt.csv"),
        "test":  str(base_path / "test/test_chatgpt.csv")
    }
    data = datasets.load_dataset("csv", data_files=data_files)

    splits_uuids = {
        "train" : data["train"]["uuid"],
        "dev" : data["dev"]["uuid"],
        "test" : data["test"]["uuid"],
    }

    data = datasets.concatenate_datasets([
        data["train"], data["dev"], data["test"]
    ])

    if is_test and args.max_samples is None:
        args.max_samples = 100

    if args.max_samples is not None:
        data = data.select(range(args.max_samples))

    data = data.shard(num_shards=world_size, index=global_rank)

    print('Loading the model...')


    if args.model_name_or_path is None:
        model_dir = Path('/leonardo_scratch/large/userexternal/gpuccett/models/hf_llama/')
        model_path = "tiny-random-llama" if is_test else "llama-2-7b-hf"
        args.model_name_or_path = str(model_dir / model_path)

    checkpoint = args.model_name_or_path

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
            tensor_parallel_size=args.tensor_parallel_size,
        )

    print('Model successfully loaded.')

    model_name = Path(checkpoint).name
    data = data.map(lambda x: get_semeval_task3_prompt(x, model_name), desc="Generating prompts")

    args.batch_size = 2 if is_test else args.batch_size
    data = data.map(lambda x:
        generate_synthetic(x, llm, params, is_test=is_test,
                           huggingface_or_vllm=huggingface_or_vllm),
        desc="Generating machine text", batched=True, batch_size=args.batch_size)

    if "output_file" not in args:
        args.output_file = "/leonardo_scratch/large/userexternal/" + \
        "gpuccett/data/PeerRead_synthetic_continuation"
    args.output_file += f"_model_{model_name}"
    args.output_file += f"_hfvllm_{huggingface_or_vllm}"

    final_dataset = save_distributed_and_collect_on_main_rank(
        data_shard=data, args=args, global_rank=global_rank,
        global_n_devices=world_size, save_after_collect=False
    )

    if global_rank == 0:
        for split, split_uuid in splits_uuids.items():
            final_dataset.filter(lambda x: x["uuid"] in split_uuid).to_pandas().to_csv(
                str(base_path / f"{split}/{split}_{model_name}.csv"), index=False
            )
            # final_dataset.filter(lambda x: x["uuid"] in split_uuid).to_json(
            #     str(base_path / f"{split}/{split}_{model_name}.jsonl")
            # )


if __name__ == "__main__":
    main(parse_args())
