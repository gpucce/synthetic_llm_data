
from argparse import ArgumentParser
import datasets

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, nargs="+")
    return parser.parse_args()



def main(args):
    for data_path in args.data_path:
        ds = datasets.load_from_disk(data_path)["train"]
        print(ds)


if __name__ == "__main__":
    main(parse_args())
