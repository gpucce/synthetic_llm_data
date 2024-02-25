
import pandas as pd
from sklearn.metrics import classification_report

def eval_rating(args):
    accuracies = {}
    for llama_model in args.data_path:
        gen_df = pd.read_csv(llama_model)
        preds = gen_df.loc[:, "machine_text"].apply(
            lambda x: int(x.startswith("sentence 1"))).to_list()
        labels = gen_df.loc[:, "first_more_abstract"].apply(int).to_list()
        print(llama_model.upper(), "N_SAMPLES", gen_df.shape[0])
        print()
        print(classification_report(labels, preds))
        out = classification_report(labels, preds, output_dict=True)
        accuracies[llama_model] = out["accuracy"]
        pd.DataFrame.from_dict(accuracies, orient="index").to_csv(
            args.output_path + "/accuracies.csv")
        print("=======================================")
