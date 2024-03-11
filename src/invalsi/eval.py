

import re
from pathlib import Path
from argparse import ArgumentParser
import datasets
import pandas as pd
from sklearn.metrics import classification_report

datasets.disable_progress_bar()

q_types = ["completa frase", "multipla", "numero", "vero/falso"]

def answer_match(x, match_threshold=35):
    ans = x["machine_text"]
    if ans is None:
        return {"answer": "None", "ans": False}

    if x["tipo"] == "multipla":
        ans_pos = ans.rfind(x['risposta'])
        if ans_pos >= 0:
            for keyword in ['risposta', 'scelta']:
                keyword_pos = (ans[max(0, ans_pos - 50):ans_pos].lower().rfind(keyword) +
                               max(0, ans_pos - 50))
                if keyword_pos >= 0 and ans_pos - keyword_pos < match_threshold:
                    # print("OK", x['risposta'])
                    # print(ans[keyword_pos:ans_pos + len(x['risposta'])])
                    # print("----------")
                    return {"answer": ans, "ans": True}
            # print("NO", x['risposta'])
            # print(ans)
            # print("----------")
            return {"answer": ans, "ans": False}
        else:
            return {"answer": ans, "ans": False}
    elif x["tipo"] == "vero/falso":
        ans = re.search("(ver[ao]|fals[ao]|\(V\)|\(F\))\.", ans, re.IGNORECASE | re.DOTALL)
        ans = ans.group(1).strip().replace("(","").replace(")", "")[0].lower() if ans else ""
        matched = ans.lower().startswith(x["risposta"].lower())
        return {"answer": ans, "ans": matched}
    elif x["tipo"] == "completa frase":
        ans = re.search(
            "(?:risposta è |risposta alla domanda è |risposta |risposta alla domanda ):?(.*)",
            ans, re.IGNORECASE | re.DOTALL)
        ans = ans.group(0).strip() if ans else ""
        matched = ans.lower().startswith(x["risposta"].lower())
        return {"answer": ans, "ans": matched}
    elif x["tipo"] == "numero":
        ans_pos = ans.rfind(x['risposta'])
        if ans_pos >= 0:
            for keyword in ['risposta', 'risultato']:
                keyword_pos = (
                    ans[max(0, ans_pos - 50):ans_pos].lower().rfind(keyword) +
                    max(0, ans_pos - 50))
                if keyword_pos >= 0 and ans_pos - keyword_pos < match_threshold:
                    # print("OK", x['risposta'])
                    # print(ans[keyword_pos:ans_pos + len(x['risposta'])])
                    # print("----------")
                    return {"answer": ans, "ans": True}
            # print("NO", x['risposta'])
            # print(ans)
            # print("----------")
            return {"answer": ans, "ans": False}
        else:
            return {"answer": ans, "ans": False}
    raise ValueError(f"Unknown answer type {x['tipo']}")

def format_match(x):
    ans = x["mixed_text"]
    if x["tipo"] == "multipla":
        ans = re.search(
            "(?:risposta è:|risposta corretta è:|risposta finale è:|risposta|risposta corretta|risposta finale)(?:è |: | è | è:)?([abcd])", # pylint: disable=line-too-long
            ans, re.IGNORECASE | re.DOTALL)
        ans = ans.group(1).replace(":", "").strip() if ans else ""
        ans = ans[0].lower() if len(ans) > 0 else ans
        matched =  ans.lower() == x["risposta"].lower()
    if x["tipo"] == "vero/falso":
        ans = re.search("(ver[ao]|fals[ao]|\(V\)|\(F\))\.", ans, re.IGNORECASE | re.DOTALL)
        ans = ans.group(1).strip().replace("(","").replace(")", "")[0].lower() if ans else ""
        matched = ans.lower().startswith(x["risposta"].lower())
    if x["tipo"] == "completa frase":
        ans = re.search(
            "(?:risposta è |risposta alla domanda è |risposta |risposta alla domanda ):?(.*)",
            ans, re.IGNORECASE | re.DOTALL)
        ans = ans.group(0).strip() if ans else ""
        matched = ans.lower().startswith(x["risposta"].lower())
    if x["tipo"] == "numero":
        ans = re.search(
            "(?:risposta finale|risposta|risposta alla domanda)(?: è | è:|: | : | :)?([\d\,]+)", 
            ans)
        ans =  ans.group(1).strip().lower() if ans else ""
        matched = ans.lower() == x["risposta"]
    return {"answer": ans, "ans": matched}

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default=".")
    return parser.parse_args()

def main(args):
    dataset = datasets.load_dataset("csv", data_files=args.data_path)
    dataset = dataset["train"]
    # dataset = dataset.map(lambda x: {"processed_ans": get_answer(x["machine_text"])})
    dataset = dataset.map(answer_match)
    output = classification_report(dataset["risposta"], dataset["answer"], output_dict=True)
    future_df = {}
    global_acc = sum(dataset['ans']) / dataset.num_rows
    print(f"Global Acc: {global_acc}")
    future_df["global_acc"] = global_acc
    for q_type in q_types:
        ds = dataset.filter(lambda x: x["tipo"] == q_type)
        local_acc = sum(ds['ans']) / ds.num_rows
        print(f"    {q_type} acc: {local_acc}")
        future_df[f"{q_type}_acc"] = local_acc

    # matching_df = pd.DataFrame.from_dict(output)
    exact_df = pd.DataFrame.from_dict(future_df, orient="index").T

    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    # matching_df.to_csv(output_path / "matching.csv")
    exact_df.to_csv(output_path / "exact.csv")

if __name__ == "__main__":
    main(parse_args())
