import os, pickle

import numpy as np
import pandas as pd

from sklearn.neighbors import KDTree

import utils, analysis_utils

utils.init()

THREADS = 12
CONFIGURATIONS = [
        # ("fashionmnist", [600, 400, 200, 100, 80, 64, 48, 32, 20]),
        # ("sift", [100, 80, 64, 48, 32, 24, 20, 18, 12]),
        ("sifttrunc", [100, 80, 64, 48, 32, 24, 20, 18, 12]),
        # ("nytimesl2", [192, 128, 96, 64, 48, 32, 26, 20, 16])
    ]

def benchmarkClassical(dataset, name, results):
    print("Benchmarking:", name)

    with open(os.path.join("classical", f"{name}.pkl"), "rb") as modelfile:
        models = pickle.load(modelfile)

    if models["idxs"] == None:
        neighbours = dataset.neighbours
    else:
        kdtree = KDTree(dataset.train.raw[models["idxs"]])
        neighbours = kdtree.query(dataset.test.raw, k=20, return_distance=False)
    
    results2 = {}
    for t in ["pca","nca","tsne"]:
        model = models[t]
        if model == None: continue

        if models["idxs"] == None:
            train_embeddings = model.transform(dataset.train.raw)
        else:
            train_embeddings = model.transform(dataset.train.raw[models["idxs"]])
        
        test_embeddings = model.transform(dataset.test.raw)
        kdtree = KDTree(train_embeddings)
        nearest = kdtree.query(test_embeddings, return_distance=False)

        results2[t] = analysis_utils.analyseResults(dataset, np.arange(len(nearest)), nearest, neighbours=neighbours)

    results[name] = results2

    return results



jobs = []
for name, dims in CONFIGURATIONS:
    for dim in dims:
        jobs.append((name, dim))

def work(job):
    name, dim = job
    dataset = utils.loadH5Dataset(utils.DATASETS[name])
    results = benchmarkClassical(dataset, f"{name}-{dim}", {})
    results = benchmarkClassical(dataset, f"{name}-truncated10-{dim}", results)
    return results

def result(result):
    results = utils.tryLoadPickle("classical_results.pkl", {})
    results.update(result)
    utils.updatePickle("classical_results.pkl", results)

analysis_utils.MultithreadedJob(jobs, work, result).run()
print("Finished!")