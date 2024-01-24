
from argparse import ArgumentParser
from datasets import load_from_disk, load_dataset, Dataset, DatasetDict


def str2bool(v):
    """Convert string to boolean"""
    if isinstance(v, bool):
        return v
    return v.lower() in ("yes", "true", "t", "1")

def custom_parse_args():
    parser = ArgumentParser()
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--col-name", type=str, default=None)
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--dataset-type", type=str, default="m4")
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--do-generation", action="store_true", default=False)
    parser.add_argument("--do-modification", action="store_true", default=False)
    parser.add_argument("--dtype", type=str, default="bfloat16",
                            choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--huggingface-or-vllm", type=str, default="huggingface")
    parser.add_argument("--human-key", type=str, default=None)
    parser.add_argument("--max-seq-len", type=int, default=150)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--model-name", type=str, default="llama-7b-chat")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--min-new-tokens", type=int, default=200)
    parser.add_argument("--multi-length-clipping", action="store_true", default=False)
    parser.add_argument("--modifier-model", type=str, default="t5-3b")
    parser.add_argument("--n-modifications", type=int, default=5)
    parser.add_argument("--n-few-shots", type=int, default=0)
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--name-or-path", type=str, default=None)
    parser.add_argument("--output-path", type=str, default="test_output")
    parser.add_argument("--padding-side", type=str, default="right")
    parser.add_argument("--preprocessing", type=str, default="xsum")
    parser.add_argument("--project", type=str, default="semeval_task_3")
    parser.add_argument("--pct-mask", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--selected-boundary", type=int, default=30)
    parser.add_argument("--split-at-random-length", action="store_true", default=False)
    parser.add_argument("--temperature", type=float, default=1.0) # HF default
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--tokenizer-path", type=str, default=None)
    parser.add_argument("--top-p", type=float, default=1.0) # HF default
    parser.add_argument("--top-k", type=int, default=50) # HF default
    parser.add_argument("--use-beam-search", type=str2bool, default=False)
    return parser.parse_args()


def plot_parse_args():
    parser = ArgumentParser()
    parser.add_argument("--col-name", type=str, default=None)
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--output-path", type=str, default="test_output")
    parser.add_argument("--split", type=str, default=None)
    return parser.parse_args()


def custom_load_dataset(args):
    if args.dataset_type == "m4":
        ds = load_dataset("json", data_files=args.data_path)

        if args.n_samples is not None:
            for split in ds:
                ds[split] = ds[split].select(range(args.n_samples))

        ds = ds["train"]

        ds = DatasetDict(
            {"train": Dataset.from_dict(
                {args.col_name: ds["machine_text"] + ds["human_text"],
                "label": [1] * len(ds["machine_text"]) + [0] * len(ds["human_text"])}
                ).shuffle(seed=args.seed)
            })

        assert len([i for i in ds["train"]["label"] if i == 1]) == \
            len([i for i in ds["train"]["label"] if i == 0])


    elif args.dataset_type == "disk":
        ds = load_from_disk(args.data_path)
        if "train" in ds:
            ds = ds["train"]
        ds = ds.shuffle(args.seed)

    else:
        ds = load_dataset(args.dataset_type, data_files=args.data_path)
        if "train" in ds:
            ds = ds["train"]
        ds = ds.shuffle(args.seed)

    return ds
