from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, help='Path to the Invalsi data')
    parser.add_argument('--output_path', type=str, help='Path to save the plot')
    return parser.parse_args()

def plot_invalsi_data(data, output_path):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=300)
    pie_data = []
    labels = []
    _type = data.loc[:, "tipo"]
    for i in _type.unique():
        labels.append(i)
        pie_data.append(len(_type[_type == i]))
    ax.pie(pie_data, labels=labels, autopct='%1.1f%%', textprops={'fontsize': 20})
    fig.savefig(output_path, bbox_inches='tight')

def main(args):
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(args.data_path)
    plot_invalsi_data(data, output_path)

if __name__ == '__main__':
    main(parse_args())
