
from pathlib import Path
from argparse import ArgumentParser
import pandas as pd

from synthetic_llm_data.src.invalsi.eval import compute_acc

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--output_path", type=str, default=".")
    return parser.parse_args()



def main(args):

    data_paths = Path(args.data_path).rglob("invalsi_mate_clean_predicted*.csv")
    future_df = {}
    for data_path in data_paths:
        data_path = str(data_path)
        data_path_name = data_path.split("_")[-1].replace(".csv", "")
        if data_path_name == "chat":
            data_path_name = " ".join(data_path.split("_")[-2:]).replace(".csv", "")
        future_df[data_path_name] = compute_acc(data_path) # .reset_index(drop=True)

    joint_results = (
        pd.concat(future_df, axis=0)
        .reset_index().drop("level_1", axis=1)
        .rename({"level_0":"model"}, axis=1).set_index("model"))

    joint_results = joint_results.sort_values("global_acc", ascending=False)
    joint_results.columns = joint_results.columns.str.replace("_acc", "")
    joint_results = (joint_results * 100).round(2)
    print(joint_results)

    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    joint_results.to_latex(output_path / "joint_results.tex", float_format="%.2f")


if __name__ == "__main__":
    main(parse_args())
