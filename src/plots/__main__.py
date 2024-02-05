
from argparse import ArgumentParser
from pathlib import Path
import json
import pandas as pd
from matplotlib.pyplot import close
from .plot_utils import plot_roc_from_dataset

def parse_args():
    parser = ArgumentParser(description="Plot ROC curve from dataset")
    parser.add_argument("--data_path", nargs="+", type=str, help="Path to dataset")
    parser.add_argument("--output_path", type=str, help="Path to save the ROC curve")
    parser.add_argument("--col_name", type=str, help="Name of column to plot")
    parser.add_argument("--detector",
        type=str, default="detectgpt", choices=["detectgpt", "supervised"])
    return parser.parse_args()

def detectpgt_plots(args):
    future_df = {"generator": [], "detector": [], "detectgpt": [], "llh": []}
    output_path = Path(args.output_path)
    for data_path in args.data_path:
        root = Path(data_path)
        for path_obj in root.glob("*"):
            if path_obj / "dataset_config.json":
                try:
                    (output_path / path_obj).mkdir(parents=True, exist_ok=True)
                    fig, detectgpt_auroc, llh_auroc = plot_roc_from_dataset(
                        path_obj, args.col_name, output_path / path_obj)
                    
                    close()
                    
                    future_df["detector"].append(str(path_obj).split("/")[-1])
                    future_df["detectgpt"].append(detectgpt_auroc)
                    future_df["llh"].append(llh_auroc)
                    future_df["generator"].append(str(data_path).split("/")[-3])
                except Exception as e:
                    print("Did not print the ROC curve for ", path_obj, " because of ", e)
                    print(path_obj)

    roc_df = pd.DataFrame.from_dict(future_df)
    # Save to formatted latex table rounded to 2 decimal places

    roc_df = roc_df.sort_values(by=["generator", "detectgpt"]) # .pivot(
        # index="generator", columns="detector", values=["detectgpt", "llh"])

    roc_df = roc_df.groupby("generator").apply(
        lambda x: x.drop("generator",axis=1).set_index("detector").T).T

    roc_df.to_csv(output_path / "auroc_table.csv", float_format="%.2f")
    # style to late with rulers
    roc_df.style.format(precision=2).to_latex(
        output_path / "auroc_table.tex", hrules=True,
        escape=True, multicol_align="l", multirow_align="c")


def supervised_plots(args):
    future_df = {"detector":[], "generator":[], "accuracy":[], "mixed_data": []}
    output_path = Path(args.output_path)
    for data_path in args.data_path:
        with open(Path(data_path) / "trainer_state.json") as jf:
            trainer_state = json.load(jf)
            logs = trainer_state["log_history"]
            final_eval_log = logs[-1]
        with open(Path(data_path) / "args.txt") as jf:
            experiment_args = json.load(jf)

        if experiment_args["real_data_path"] is not None:
            future_df["mixed_data"].append("mixed")
        else:
            future_df["mixed_data"].append("in_domain")

        future_df["detector"].append(experiment_args["model_name_or_path"].split("/")[-1])
        future_df["generator"].append(experiment_args["data_path"].split("/")[-2])
        future_df["accuracy"].append(final_eval_log["eval_accuracy"])

    acc_df = pd.DataFrame.from_dict(future_df)
    acc_df.to_csv(output_path / "accuracy_table.csv")
    acc_df.to_latex(
        output_path / "accuracy_table.tex", float_format="%.2f", index=False, escape=True)

    return acc_df

def main(args):
    if args.detector == "detectgpt":
        detectpgt_plots(args)
    elif args.detector == "supervised":
        supervised_plots(args)
    else:
        raise ValueError(f"Detector {args.detector} not recognized")

if __name__ == "__main__":
    main(parse_args())
