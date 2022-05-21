import os
import pandas as pd
from itertools import product

if __name__ == '__main__':
    idx = os.listdir("../data/High_Resolution_compressed/")[:-1]

    luminosity = pd.Series(list(range(1, 31)), dtype=str).apply(lambda x: 'L'+x).tolist()
    emotion = pd.Series(["01", "02"]).apply(lambda x: 'E'+x).tolist()
    view_point = pd.Series(list(range(5, 10)), dtype=str).apply(lambda x: 'C'+x).tolist()

    keys = [idx, ['S001'], luminosity, emotion, view_point]
    names = list(product(*keys))

    names = pd.DataFrame(names)
    names['path'] = names.agg(lambda x: f"{x[0]}/{x[1]}/{x[2]}/{x[3]}/{x[4]}", axis=1)
    names = names[[0, "path"]]

    names.to_csv('../data/info.csv')
