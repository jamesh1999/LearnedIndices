import os, argparse, datetime

import pandas as pd
import pytorch_lightning as pl

import utils

if __name__ == "__main__":
    # ------------------------
    # Read arguments
    # ------------------------
    parser = argparse.ArgumentParser(
        prog="ANN Trainer"
    )

    parser.add_argument("-d", "--dimensions", type=int)
    parser.add_argument("-D", "--dataset", choices=utils.DATASETS.keys())
    parser.add_argument("-m", "--model", choices=utils.MODELS.keys())
    parser.add_argument("-t", "--type", choices=utils.TYPES.keys())

    args = parser.parse_args()

    # ------------------------
    # Prepare Datasets / Loaders / Loggers
    # ------------------------

    dataset = utils.loadH5Dataset(utils.DATASETS[args.dataset])
    inputs = dataset.test.raw[0].size

    train_loader   = utils.createDataLoader(dataset.train, shuffle=True)
    test_loader    = utils.createDataLoader(dataset.test)

    logger = pl.loggers.CSVLogger("")

    # ------------------------
    # Create & Train Model
    # ------------------------

    model = utils.TYPES[args.type](utils.MODELS[args.model], inputs=inputs, dimensions=args.dimensions)
    trainer = pl.Trainer(logger=logger, **utils.TRAINER_ARGS)
    if args.type != "identity":
        trainer.fit(model, train_loader, test_loader)
    else:
        trainer.predict(model, test_loader)

    # ------------------------
    # Save Model & Parameters
    # ------------------------
    params = vars(args)
    params["time"] = datetime.datetime.now()
    df = pd.DataFrame([params])
    try:
        prev_df = pd.read_pickle("hyperparameters.pkl")
        df = pd.concat([prev_df, df], ignore_index=True)
    except FileNotFoundError:
        pass
    df.to_pickle("hyperparameters.pkl")

    trainer.save_checkpoint(os.path.join("checkpoints", f"{df.index.max()}.ckpt"))
