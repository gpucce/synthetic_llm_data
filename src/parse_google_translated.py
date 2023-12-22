import re
from itertools import pairwise

HUMAN_PATTERN = "\*\*\*---\*\*\*---\*\*\*"
MACHINE_PATTERN = "**-**-**"

def parse_google_translated(data_path):
    with open(data_path) as dp:
        data = dp.read()

    data = re.split(f"{HUMAN_PATTERN} \[(\d+)\]", data)[1:]
    data = [(i[0],) + tuple(j for j in i[1].split(MACHINE_PATTERN))
            for i in list(pairwise(data))[::2]]
    return data
