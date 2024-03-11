from argparse import ArgumentParser
from pathlib import Path
import random
import pandas as pd
from sklearn.metrics import classification_report
from scipy.stats import pearsonr

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    return parser.parse_args()

def get_lmrater_accuray(data_path, concept):
    gen_df = pd.read_csv(data_path)
    preds = gen_df.loc[:, "machine_text"].apply(lambda x: int(x.startswith("sentence 1"))).to_list()
    label_key = "first_more_abstract" if concept == "abstraction" else "first_more_inclusive"
    labels = gen_df.loc[:, label_key].apply(int).to_list()
    return classification_report(labels, preds, output_dict=True)["accuracy"]

def get_rank(x):
    try:
        return int(x.split("the rank is: ")[1][:1])
    except:
        return random.randint(1, 5)

def get_lmranker_correlation(data_path, concept):
    df = pd.read_csv(data_path)
    # df["preds"] = df["mixed_text"].apply(lambda x: int(x.split("the rank is: ")[-1][:1]))
    

    df["preds"] = df["machine_text"].apply(get_rank)
    label_key = "abs_mean" if concept == "abstraction" else "inc_mean"
    df["labels"] = df[label_key]
    return pearsonr(df["labels"], df["preds"]).correlation

def main(args):
    data_path = Path(args.data_path)
    future_df = {"choice": [], "n_shots": [], "experiment_kind": [], "name": [], "score": []}
    for concept in ["abstraction", "inclusiveness"]:
        path = Path(data_path / concept)
        experiments = path.glob("**/*.csv")
        for experiment in experiments:
            
            experiment_name = str(experiment)
            choice, n_shots, experiment_kind, name = experiment_name.split("/")[2:6]
            print(experiment, experiment_kind)
            name = name.split("_")[-1].replace(".csv", "").replace("instruct", "mistral")
            if "rater" in experiment_kind:
                future_df["choice"].append(choice)
                future_df["n_shots"].append(n_shots)
                future_df["experiment_kind"].append(experiment_kind)
                future_df["score"].append(get_lmrater_accuray(experiment, concept))
                future_df["name"].append(name)
            elif "ranker" in experiment_kind:
                future_df["choice"].append(choice)
                future_df["n_shots"].append(n_shots)
                future_df["experiment_kind"].append(experiment_kind)
                future_df["score"].append(get_lmranker_correlation(experiment, concept))
                future_df["name"].append(name)


    df = pd.DataFrame.from_dict(future_df)

    ranker_df = df.loc[df.loc[:, "experiment_kind"] == "ranker_experiment", :]
    ranker_df = ranker_df.sort_values(by=["n_shots", "experiment_kind", "name"]).pivot(
        index=["experiment_kind", "choice", "n_shots",], columns="name", values="score").map(
            lambda x: round(x, 2) if not pd.isna(x) else x)
    ranker_df.to_latex(data_path / "scores_ranker.tex", escape=True)
    ranker_df.to_csv(data_path / "scores_ranker.csv", index=True)

    rater_df = df.loc[df.loc[:, "experiment_kind"] == "rater_experiment", :]
    rater_df = rater_df.sort_values(by=["n_shots", "experiment_kind", "name"]).pivot(
        index=["experiment_kind", "choice", "n_shots"], columns="name", values="score").map(
            lambda x: round(x, 2) if not pd.isna(x) else x)
    rater_df.to_latex(data_path / "scores_rater.tex", escape=True)
    rater_df.to_csv(data_path / "scores_rater.csv", index=True)    

if __name__ == "__main__":
    main(parse_args())