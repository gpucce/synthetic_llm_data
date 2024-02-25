

import re
from argparse import ArgumentParser
import datasets

from sklearn.metrics import classification_report

datasets.disable_progress_bar()

q_types = ["completa frase", "multipla", "numero", "vero/falso"]

def get_answer(x):
    try:
        ans = re.search("CONCLUSIONE:(.{1,20})(==|\n)", x)
        if ans:
            return ans.group(1).strip().replace("{", "").replace("}", "").replace("==", "")
    except:
        pass
    return ""

def answered(x):
    ans = get_answer(x["machine_text"])
    print(ans, "|||", x["risposta"], "|||", x["machine_text"])
    if x["tipo"] in ["multipla", "numero"]:
        return ans != "" and ans.lower().startswith(x["risposta"].lower())
    if x["tipo"] == "vero/falso":
        return ans != "" and ans.lower()[0] == x["risposta"].lower()[0]
    if x["tipo"] == "completa frase":
        return ans.lower().startswith(x["risposta"].lower())
    return False

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    return parser.parse_args()

def main(args):
    dataset = datasets.load_dataset("csv", data_files=args.data_path)
    dataset = dataset["train"]
    dataset = dataset.map(lambda x: {"processed_ans": get_answer(x["machine_text"])})
    dataset = dataset.map(lambda x: {"ans": answered(x)})
    # print(classification_report(dataset["ans"], [True] * dataset.num_rows))
    print(f"Global Acc: {sum(dataset['ans']) / dataset.num_rows} F1: ")
    for q_type in q_types:
        ds = dataset.filter(lambda x: x["tipo"] == q_type)
        print(f"    {q_type} acc: {sum(ds['ans']) / ds.num_rows}")

if __name__ == "__main__":
    main(parse_args())
