from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import numpy as np
import pandas as pd
import scipy.stats


def show_tsne(embs, labels):
    coor = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(embs)
    plt.figure()
    plt.scatter(coor[:, 0], coor[:, 1], c=labels, s=5)
    plt.show()


def show_sample(good_idx, info, embs):
    for i in good_idx[:5]:
        bboxing(f"../data/High_Resolution_compressed/{info.iloc[i, 1]}.jpg", embs[i][0].bbox)


def bboxing(path, bbox):
    img = Image.open(path).convert('RGB')
    draw = ImageDraw.Draw(img)
    draw.rectangle(bbox, outline=(0, 255, 0), width=3)
    img.show()


def show_dam_hist(soles, dups):
    # assert len(soles) != 0
    # assert len(dups) != 0
    soles = pd.Series(soles).dropna()
    dups = pd.Series(dups).dropna()
    print(f"T test: {scipy.stats.ttest_ind(soles.values, dups.values)}")

    print(f"Different : {len(soles)} / Same : {len(dups)}")
    plt.figure()
    plt.hist(soles.tolist(), stacked=True, density=True, color='r', label='Different', alpha=0.3)
    plt.hist(dups.tolist(), stacked=True, density=True, color='b', label='Same', alpha=0.7)
    plt.legend()
    plt.show()
    print()


def show_cosig_plot(sole_cosig, dup_cosig):
    plt.figure()
    plt.scatter(sole_cosig, np.zeros_like(sole_cosig), s=2, label='Different')
    plt.scatter(dup_cosig, np.ones_like(dup_cosig), s=2, label='Same')
    plt.legend()
    plt.show()
