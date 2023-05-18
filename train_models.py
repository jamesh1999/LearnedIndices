import os, argparse, datetime

import pandas as pd
import pytorch_lightning as pl

import utils

if __name__ == "__main__":
    utils.init()

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

    parser.add_argument("--crecon", type=float, default=1)
    parser.add_argument("--calpha", type=float, default=1)
    parser.add_argument("--clambda", type=float, default=3000)
    parser.add_argument("--cortho", type=float, default=0.00001)
    parser.add_argument("--cspace", type=float, default=1)
    parser.add_argument("--ctriplet", type=float, default=1)

    parser.add_argument("--triplet_reps", type=int, default=1)
    parser.add_argument("--encoder_wd", type=float, default=0)
    parser.add_argument("--decoder_wd", type=float, default=0)

    parser.add_argument("--power", type=float, default=2)

    parser.add_argument("--optimiser", choices=["adam","sgd"], default="adam")
    
    args = parser.parse_args()

    # ------------------------
    # Load & Update DataFrame
    # ------------------------
    params = vars(args)
    params["time"] = datetime.datetime.now()
    df = pd.DataFrame([params])
    try:
        prev_df = pd.read_pickle("hyperparameters.pkl")
        df = pd.concat([prev_df, df], ignore_index=True)
    except FileNotFoundError:
        pass

    # ------------------------
    # Prepare Datasets / Loaders / Loggers
    # ------------------------
    dataset = utils.loadH5Dataset(utils.DATASETS[args.dataset])
    inputs = dataset.test.raw[0].size

    train_loader   = utils.createDataLoader(dataset.train, shuffle=True)
    test_loader    = utils.createDataLoader(dataset.test)

    logger = pl.loggers.CSVLogger("", "logging", df.index.max())

    # ------------------------
    # Create & Train Model
    # ------------------------
    constants = {
        "dimensions":args.dimensions,
        "crecon":args.crecon,
        "calpha":args.calpha,
        "clambda":args.clambda,
        "cortho":args.cortho,
        "cspace":args.cspace,
        "ctriplet":args.ctriplet,
        "encoder_wd":args.encoder_wd,
        "decoder_wd":args.decoder_wd,
        "optimiser":args.optimiser,
        "power":args.power,
        "triplet_repetitions":args.triplet_reps
    }

    fdist = utils.DISTANCES[args.dataset]

    model = utils.TYPES[args.type](
        utils.MODELS[args.model],
        inputs=inputs,
        width=1000,
        inner_dist=None,
        outer_dist=fdist,
        **constants
    )
    trainer = pl.Trainer(logger=logger, **utils.TRAINER_ARGS)
    if args.type != "identity":
        trainer.fit(model, train_loader, test_loader)
    else:
        trainer.predict(model, test_loader)

    # ------------------------
    # Save Model & Parameters
    # ------------------------
    df.to_pickle("hyperparameters.pkl")
    trainer.save_checkpoint(os.path.join("checkpoints", f"{df.index.max()}.ckpt"))
