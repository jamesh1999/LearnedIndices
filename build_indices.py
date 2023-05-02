import os, argparse, datetime, pickle

import torch
import pandas as pd
import pytorch_lightning as pl

from sklearn.neighbors import KDTree
import pynndescent
import hnswlib

import utils, models

if __name__ == "__main__":
    # ------------------------
    # Read arguments
    # ------------------------
    parser = argparse.ArgumentParser(
        prog="ANN Index Builder"
    )

    parser.add_argument("-I", "--index", choices=["kdtree", "pynn", "hnsw"])
    parser.add_argument("-i", "--index-param", type=int)
    parser.add_argument("-m", "--reduction-model", type=int)

    args = parser.parse_args()

    # ------------------------
    # Prepare Datasets / Loaders / Loggers
    # ------------------------

    with open("hyperparameters.pkl", "rb") as file:
        df = pd.read_pickle(file)

    dataset = utils.loadH5Dataset(utils.DATASETS[df.iloc[args.reduction_model]["dataset"]])
    train_loader = utils.createDataLoader(dataset.train)

    # ------------------------
    # Load & Apply Model
    # ------------------------

    model = utils.loadModel(df, args.reduction_model)
    trainer = pl.Trainer(logger=False, **utils.TRAINER_ARGS)
    train_encodings = utils.numpyConvert(torch.cat(trainer.predict(model, train_loader)))
    dimensions = train_encodings[0].size

    # ------------------------
    # Build Index
    # ------------------------

    if args.index == "kdtree":
        idx = KDTree(train_encodings)
    elif args.index == "pynn":
        idx = pynndescent.NNDescent(train_encodings, n_neighbors=args.index_param)
    elif args.index == "hnsw":
        idx = hnswlib.Index(space='l2', dim=dimensions)
        idx.init_index(max_elements=len(train_encodings), ef_construction=args.index_param, M=dimensions)
        idx.set_num_threads(4)
        idx.add_items(train_encodings)

    # ------------------------
    # Save Index & Parameters
    # ------------------------

    params = vars(args)
    params["time"] = datetime.datetime.now()
    df = pd.DataFrame([params])
    try:
        prev_df = pd.read_pickle("indices.pkl")
        df = pd.concat([prev_df, df], ignore_index=True)
    except FileNotFoundError:
        pass
    df.to_pickle("indices.pkl")

    with open(os.path.join("indices", f"{df.index.max()}.pkl"), "wb") as output:
        pickle.dump(idx, output)
