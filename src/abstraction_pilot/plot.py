from argparse import ArgumentParser
from pathlib import Path
import random
import pandas as pd
from sklearn.metrics import classification_report
from scipy.stats import pearsonr, spearmanr

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--lang", type=str, default="eng", choices=["eng", "ita"])
    return parser.parse_args()

def get_lmrater_accuray(data_path, concept):
    gen_df = pd.read_csv(data_path)
    preds = gen_df.loc[:, "machine_text"].apply(lambda x: int(x.startswith("sentence 1"))).to_list()
    label_key = "first_more_abstract" if concept == "abstraction" else "first_more_inclusive"
    labels = gen_df.loc[:, label_key].apply(int).to_list()
    return classification_report(labels, preds, output_dict=True)["accuracy"]

def get_rank(x, args):
    splitter = "the rank is" if args.lang == "eng" else "il livello è"
    if args.lang == "ita" and splitter not in x.lower():
        splitter = "il punteggio è"
    try:
        return int(x.lower().split(splitter)[1].replace(":", "").strip()[0])
    except Exception as e:
        print(e)
        return random.randint(1, 5)

def get_lmranker_correlation(data_path, concept, args):
    df = pd.read_csv(data_path)

    df["preds"] = df["machine_text"].apply(lambda x: get_rank(x, args))
    label_key = "abs_mean" if concept == "abstraction" else "inc_mean"
    df["labels"] = df[label_key]
    pearson = pearsonr(df["labels"], df["preds"]).correlation
    spearman = spearmanr(df["labels"], df["preds"]).correlation
    return pearson, spearman

def main(args):
    data_path = Path(args.data_path)
    future_df = {"choice": [], "n_shots": [], "experiment_kind": [],
                 "name": [], "score": [], "pearson": [], "spearman": []}
    for concept in ["abstraction", "inclusiveness"]:
        path = Path(data_path / concept)
        experiments = path.glob("**/**/*.csv")
        experiments = [i for i in experiments if "ipynb" not in str(i)]
        if args.lang == "ita":
            experiments = [i for i in experiments if "ita" in str(i)]
        else:
            experiments = [i for i in experiments if "ita" not in str(i)]
        for experiment in experiments:
            experiment_name = str(experiment)
            choice, n_shots, experiment_kind, name = experiment_name.rsplit("/")[-4:]
            print(experiment, experiment_kind, n_shots, choice)
            n_shots = int(n_shots.split("_")[0])
            name = name.split("_")[-1].replace(".csv", "").replace("instruct", "mistral")
            if "rater" in experiment_kind:
                future_df["choice"].append(choice)
                future_df["n_shots"].append(n_shots)
                future_df["experiment_kind"].append(experiment_kind)
                future_df["score"].append(get_lmrater_accuray(experiment, concept))
                future_df["pearson"].append(None)
                future_df["spearman"].append(None)
                future_df["name"].append(name)
            elif "ranker" in experiment_kind:
                future_df["choice"].append(choice)
                future_df["n_shots"].append(n_shots)
                future_df["experiment_kind"].append(experiment_kind)
                pearson, spearman = get_lmranker_correlation(experiment, concept, args)
                future_df["name"].append(name)
                future_df["pearson"].append(pearson)
                future_df["spearman"].append(spearman)
                future_df["score"].append(None)

    print(future_df["choice"])
                
    df = pd.DataFrame.from_dict(future_df)
    if df.shape[0] == 0:
        print("NO EXPERIMENTS FOUND")
        return

    ranker_df = df.loc[df.loc[:, "experiment_kind"] == "ranker_experiment", :]
    ranker_df = ranker_df.sort_values(by=["n_shots", "experiment_kind", "name"]).pivot(
        index=["experiment_kind", "choice", "n_shots",],
        columns="name", values=["pearson", "spearman"]).applymap(
            lambda x: round(x, 2) if not pd.isna(x) else x)

    latex_ranker_df = (ranker_df.reset_index()
        .drop(["experiment_kind"], axis=1)
        .rename({"n_shots": "n shots"}, axis=1)
        .replace({"3_shots": "3 shots", "0_shots": "0 shots"})
        .set_index(["choice", "n shots"]))

    latex_ranker_df.to_latex(
        data_path / "scores_ranker.tex", escape=True, float_format="%.2f", multicolumn_format="c")
    latex_ranker_df.style.format(
        "{:.2f}").to_latex(data_path / f"scores_ranker_{args.lang}.tex", hrules=True,
        multicol_align="c",  # position="ht",
        multirow_align="c", column_format="ll" + "c" * len(latex_ranker_df.columns))

    ranker_df.to_csv(data_path / f"scores_ranker_{args.lang}.csv", index=True)

    rater_df = df.loc[df.loc[:, "experiment_kind"] == "rater_experiment", :]
    rater_df = rater_df.sort_values(by=["n_shots", "experiment_kind", "name"]).pivot(
        index=["experiment_kind", "choice", "n_shots"], columns="name", values="score").applymap(
            lambda x: round(x, 2) if not pd.isna(x) else x)
    (rater_df.reset_index().drop("experiment_kind", axis=1).rename({"n_shots": "n shots"}, axis=1)
        .set_index(["choice", "n shots"])
        .to_latex(data_path / f"scores_rater_{args.lang}.tex", escape=True, 
                  float_format="%.2f", multicolumn_format="c"))
    rater_df.to_csv(data_path / f"scores_rater_{args.lang}.csv", index=True)

if __name__ == "__main__":
    main(parse_args())
