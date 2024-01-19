import json
from pathlib import Path
from argparse import ArgumentParser
from datasets import load_dataset

from synthetic_llm_data.src.data_generation.data_complete import generation

from .args import lmrater_parse_args



def main(args):

    print("Loading data...")
    ds = load_dataset("csv", data_files=args.data_path)["train"]

    if args.max_samples is not None:
        ds = ds.shuffle(args.seed).select(range(args.max_samples))

    ds = generation(args, ds)

    ds.to_csv(args.output_path)

    with open(Path(args.output_path).parent / "experiment_params.json", "w") as f:
        f.write(json.dumps(vars(args)))

if __name__ == "__main__":
    main(lmrater_parse_args())
