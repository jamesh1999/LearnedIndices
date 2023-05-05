import torch
import pytorch_lightning as pl

import numpy as np

from sklearn.neighbors import KDTree

import utils



def analyseResults(dataset : utils.H5Loader, queries, results : np.ndarray):
    errors = []
    failed = 0
    for q,rs in zip(queries, results):

        if results.ndim == 1:
            r = rs
        else:
            qraw = dataset.test.raw[q]
            rraws = dataset.train.raw[rs]
            distances = np.sum(np.square(rraws - qraw), 1)
            r = rs[np.argmin(distances)]

        x = np.argmax(dataset.neighbours[q] == r)
        if dataset.neighbours[q][x] == r:
            errors.append(x)
        else:
            failed += 1
    errors = np.array(errors)

    return {
        "mean":np.mean(errors),
        "std":np.std(errors),
        "acc":100*np.mean(errors == 0),
        "fail":100*(failed / len(results)),
        "raw":errors
    }

def benchmarkModel(df, idx, dataset=None, query_count=99999999, shuffle=False):
    if dataset == None:
        dataset = utils.loadH5Dataset(utils.DATASETS[df.iloc[idx]["dataset"]])

    model = utils.loadModel(df, idx)
    
    # Generate embeddings
    trainer = pl.Trainer(logger=False, **utils.TRAINER_ARGS)
    train_loader = utils.createDataLoader(dataset.train)
    train_embeddings = utils.numpyConvert(torch.cat(trainer.predict(model, train_loader)))
    test_loader = utils.createDataLoader(dataset.test)
    test_embeddings = utils.numpyConvert(torch.cat(trainer.predict(model, test_loader)))

    query_count = min(query_count, len(test_embeddings))
    query_idxs = np.arange(len(test_embeddings))
    if shuffle:
        np.random.shuffle(query_idxs)
    query_idxs = query_idxs[:query_count]

    if True: # Use KD-tree
        kdtree = KDTree(train_embeddings)
        nearest = kdtree.query(test_embeddings[query_idxs], return_distance=False)

    else: # Approximate by just testing with 100 nearest
        nearest = []
        for idx in query_idxs:
            deltas = train_embeddings[dataset.neighbours[idx]] - test_embeddings[idx]
            deltas = np.sum(np.square(deltas), 1)
            nearest.append([dataset.neighbours[np.argmin(deltas)]])
        nearest = np.array(nearest)

    return query_idxs, nearest