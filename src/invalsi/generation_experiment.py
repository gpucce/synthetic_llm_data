import json
from pathlib import Path

import torch
from datasets import load_dataset

from synthetic_llm_data.src.data_generation.data_complete import generation
from synthetic_llm_data.src.data_generation.args import get_base_parser

def invalsi_parse_args():
    parser = get_base_parser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument(
        "--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--project", type=str, default="wemb")
    parser.add_argument("--selected_boundary", type=int, default=100000)
    parser.add_argument("--split_at_random_length", action="store_true", default=False)
    return parser.parse_args()

def main(args):

    torch.manual_seed(args.seed)

    print("Loading data...")
    ds = load_dataset("csv", data_files=args.data_path)["train"]

    if args.max_prompts is not None:
        ds = ds.shuffle(args.seed).select(range(args.max_prompts))

    ds = generation(args, ds)
    ds = ds.map(lambda x: {"answer": x["mixed_text"].split("==CONCLUSIONE:")[1].split("==")[0]})
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    ds.to_csv(args.output_path)

    with open(Path(args.output_path).parent / "lmranker_experiment_params.json", "w") as f:
        f.write(json.dumps(vars(args)))

if __name__ == "__main__":
    main(invalsi_parse_args())
