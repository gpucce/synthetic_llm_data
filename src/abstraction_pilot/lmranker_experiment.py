import json
from pathlib import Path

import torch
from datasets import load_dataset

from synthetic_llm_data.src.data_generation.data_complete import generation
from .args import lmrater_parse_args

def main(args):

    torch.manual_seed(args.seed)

    print("Loading data...")
    ds = load_dataset("csv", data_files=args.data_path)["train"]

    if args.max_prompts is not None:
        ds = ds.shuffle(args.seed).select(range(args.max_prompts))

    ds = generation(args, ds)

    ds = ds.map(lambda x: {"preds": int(x["mixed_text"].split("the rank is: ")[-1][:1])})
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    ds.to_csv(args.output_path)

    with open(Path(args.output_path).parent / "lmranker_experiment_params.json", "w") as f:
        f.write(json.dumps(vars(args)))

if __name__ == "__main__":
    main(lmrater_parse_args())
