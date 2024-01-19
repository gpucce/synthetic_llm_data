
import re
import random
from itertools import combinations

import pandas as pd

from synthetic_llm_data.src.synthetic_detection.detect_gpt.utils import (
    tokenize_and_mask_with_indices
)

def underline_replacement(x, col_name):
    y = x["text"]
    z = x[col_name]
    x1 = x["begin"]
    x2 = x["end"]
    end = -(len(y) - x2 - 1)
    return {
        f"{col_name}_underlined_extraction": z[:x1] + "||| " + z[x1:end] + " |||" + z[end:],
        f"{col_name}_target_token": z[x1:end],
        f"{col_name}_begin": x1,
        f"{col_name}_end": end
    }

def postprocess_data(data):

    data = data.map(lambda x: {"extracted_target": x["text"][x["begin"]:x["end"]+1]})
    data = data.map(lambda x:
        {"masked": tokenize_and_mask_with_indices(x["text"], x["begin"], x["end"])})
    synthetic_cols = [col for col in data.column_names if "synthetic" in col]
    for col in synthetic_cols:
        data = data.map(lambda x: underline_replacement(x, col))
    return data

def blow_columns(x, col_name="text"):
    future_df = {}
    synthetic_cols = [col for col in x.columns if re.match("text_synthetic_\d+$", col)]
    for c in ["ID", "text", "begin", "end", "target_token"]:
        future_df[c] = list(x[c])

    for n_replica, _ in enumerate(synthetic_cols):
        print(n_replica)
        future_df["ID"].extend([i + f"_repl_{n_replica}" for i in x["ID"]])
        future_df["text"].extend(x[f"{col_name}_synthetic_{n_replica}"])
        future_df["begin"].extend(i for i in x[f"{col_name}_synthetic_{n_replica}_begin"])
        future_df["end"].extend(i for i in x[f"{col_name}_synthetic_{n_replica}_end"])
        future_df["target_token"].extend(
            i for i in x[f"{col_name}_synthetic_{n_replica}_target_token"])

    for c in x.columns:
        if c not in synthetic_cols and c not in future_df:
            future_df[c] = list(x[c])
            for n_replica, _ in enumerate(synthetic_cols):
                future_df[c].extend(["" for _ in x[c]])

    return pd.DataFrame.from_dict(future_df)

def get_pairs(df):
    df["ID_GROUP"] = df["ID"].apply(lambda x: x.split("_")[0])
    gdf = df.groupby("ID_GROUP")
    def _get_pairs(x):
        if x.shape[0] == 1:
            return x

        future_df = {"ID": [], "text": [], "text1": [], "text2": [], "target_token": [],
            "first_more_abstract": [], "first_more_inclusive": []}

        for i,j in combinations(range(x.shape[0]), 2):
            if random.random() < 0.5:
                j, i = i, j

            future_df["text"].append(x["text"].iloc[i] + " " + x["text"].iloc[j])
            future_df["text1"].append(x["text"].iloc[i])
            future_df["text2"].append(x["text"].iloc[j])
            future_df["ID"].append(x["ID"].iloc[i] + "_" + x["ID"].iloc[j])
            future_df["target_token"].append(
                x["target_token"].iloc[i] + "_" + x["target_token"].iloc[j])
            future_df["first_more_abstract"].append(x["abs_mean"].iloc[i] > x["abs_mean"].iloc[j])
            future_df["first_more_inclusive"].append(x["inc_mean"].iloc[i] > x["inc_mean"].iloc[j])

        return pd.DataFrame(future_df)

    new_df = gdf.apply(_get_pairs)
    # new_df.drop("ID_GROUP", axis=1, inplace=True)
    return new_df
