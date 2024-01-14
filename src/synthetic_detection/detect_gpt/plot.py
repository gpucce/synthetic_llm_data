
from pathlib import Path
import datasets
from sklearn.metrics import roc_curve
from .utils import plot_roc, plot_parse_args


def main(args):
    ds = datasets.load_from_disk(args.data_path)
    if args.split is not None:
        ds = ds[args.split]
    col_name = args.col_name
    fpr, tpr, _ = roc_curve(ds["label"], [-i for i in ds[f"{col_name}_loss"]])
    fig, _ = plot_roc(fpr, tpr)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(str(output_path), bbox_inches="tight")


if __name__ == "__main__":
    main(plot_parse_args())
