
from argparse import ArgumentParser
from datasets import load_from_disk, load_dataset, Dataset, DatasetDict

def custom_parse_args():
    parser = ArgumentParser()
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--col-names", type=str, default=None)
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--dataset-type", type=str, default="m4")
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--do-generation", action="store_true", default=False)
    parser.add_argument("--huggingface-or-vllm", type=str, default="huggingface")
    parser.add_argument("--human-key", type=str, default=None)
    parser.add_argument("--max-seq-len", type=int, default=150)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--model-name", type=str, default="llama-7b-chat")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--min-new-tokens", type=int, default=200)
    parser.add_argument("--modifier-model", type=str, default="t5-3b")
    parser.add_argument("--n-modifications", type=int, default=5)
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--name-or-path", type=str, default=None)
    parser.add_argument("--output-path", type=str, default="test_output")
    parser.add_argument("--preprocessing", type=str, default="xsum")
    parser.add_argument("--pct-mask", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-at-random-length", action="store_true", default=False)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--tokenizer-path", type=str, default=None)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=10)
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
        ds = ds["train"]
        ds = ds.shuffle(args.seed).select(range(args.n_samples))

    else:
        ds = load_dataset(args.dataset_type, data_files=args.data_path)
        ds = ds["train"]
        ds = ds.shuffle(args.seed).select(range(args.n_samples))

    return ds
