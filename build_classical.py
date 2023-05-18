import time, pickle, os

import torch
from torchvision import datasets
import pytorch_lightning as pl

import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.neighbors import KDTree, NeighborhoodComponentsAnalysis
from sklearn.manifold import TSNE

import pandas as pd

import models, utils, analysis_utils

utils.init()

def buildClassicalMethods(name, xs, dimensions, idxs=None):
    if isinstance(idxs, np.ndarray):
        xs = xs[idxs]
        idxs = idxs.tolist()
    
    ys = np.ones(len(xs))

    print(f"Fitting {name} PCA")
    pca = PCA(n_components=dimensions)
    pca.fit(xs)

    nca = None
    tsne = None
    if len(xs) <= 10_000: # Only for small datasets [O(n^2) memory]
        print(f"Fitting {name} NCA")
        nca = NeighborhoodComponentsAnalysis(n_components=dimensions)
        nca.fit(xs, ys)

    if len(xs) <= 2_000: # Only for small datasets [O(n^2) memory]
        print(f"Fitting {name} tSNE")
        tsne = TSNE(n_components=dimensions, method="exact")
        tsne.fit(xs)

    with open(os.path.join("classical", f"{name}.pkl"), "wb") as output:
        pickle.dump({"pca":pca, "nca":nca, "tsne":tsne, "idxs":idxs}, output)



jobs = []
for name, dims in [
        # ("fashionmnist", [600, 400, 200, 100, 80, 64, 48, 32, 20]),
        # ("sift", [100, 80, 64, 48, 32, 24, 20, 18, 12]),
        ("sifttrunc", [100, 80, 64, 48, 32, 24, 20, 18, 12]),
        # ("nytimesl2", [192, 128, 96, 64, 48, 32, 26, 20, 16])
    ]:
    for d in dims:
        jobs.append((name, d))

def work(job):
    name, d = job
    dataset = utils.loadH5Dataset(utils.DATASETS[name]).train.raw
    perm = np.arange(len(dataset))
    np.random.shuffle(perm)
    idxs10 = perm[:10_000]
    idxs2 = perm[:2_000]

    # Full size
    buildClassicalMethods(f"{name}-{d}", dataset, d)

    # Truncated to 10k
    buildClassicalMethods(f"{name}-truncated10-{d}", dataset, d, idxs=idxs10)

    # Truncated to 2k
    # buildClassicalMethods(f"{name}-truncated2-{d}", dataset, d, idxs=idxs2)

def result(result):
    pass

#analysis_utils.MultithreadedJob(jobs, work, result).run()
for j in jobs:
    work(j)