import os, argparse, datetime, copy

import pandas as pd
import pytorch_lightning as pl

import optuna

import utils

model = None
best_model = None

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
    parser.add_argument("--cspace", type=float, default=1)
    parser.add_argument("--ctriplet", type=float, default=1)

    parser.add_argument("--triplet_reps", type=int, default=1)

    parser.add_argument("--power", type=float, default=2)
    
    args = parser.parse_args()

    # ------------------------
    # Prepare Datasets / Loaders / Loggers
    # ------------------------
    dataset = utils.loadH5Dataset(utils.DATASETS[args.dataset])
    inputs = dataset.test.raw[0].size

    train_loader   = utils.createDataLoader(dataset.train, shuffle=True)
    test_loader    = utils.createDataLoader(dataset.test)

    # ------------------------
    # Find Starting Point
    # ------------------------
    constants = {
        "dimensions":args.dimensions,
        "crecon":args.crecon,
        "calpha":args.calpha,
        "clambda":args.clambda,
        "cspace":args.cspace,
        "ctriplet":args.ctriplet,
        "power":args.power,
        "triplet_repetitions":args.triplet_reps
    }

    pretrainer_args = copy.deepcopy(utils.TRAINER_ARGS)
    pretrainer_args["max_epochs"] = 15

    model = utils.TYPES[args.type](
        utils.MODELS[args.model],
        inputs=inputs,
        dimensions=constants["dimensions"],
        width=1000,
        crecon=1
    )
    
    trainer = pl.Trainer(
        logger=False,
        **pretrainer_args
    )
    trainer.fit(model, train_loader, test_loader)
    lossr = trainer.validate(model, test_loader)[-1]["test_loss_recon"]

    model = utils.TYPES[args.type](
        utils.MODELS[args.model],
        inputs=inputs,
        dimensions=constants["dimensions"],
        width=1000,
        cspace=1,
        power=constants["power"],
        triplet_repetitions=constants["triplet_repetitions"]
    )

    trainer = pl.Trainer(
        logger=False,
        **pretrainer_args
    )
    trainer.fit(model, train_loader, test_loader)
    lossspace = trainer.validate(model, test_loader)[-1]["test_loss_space"]

    base = lossr/lossspace

    # ------------------------
    # Find Optimal Loss
    # ------------------------
    def objective(trial):
        global model

        trainer_args = copy.deepcopy(utils.TRAINER_ARGS)
        space = trial.suggest_float("space", 1e-3, 1, log=True)
        trainer_args["callbacks"].append(utils.PyTorchLightningPruningCallbackLatest(trial, monitor="test_loss_space"))
        logger = False#utils.DFLogger(trial.number)

        constants["cspace"] = space*base
        model = utils.TYPES[args.type](
            utils.MODELS[args.model],
            inputs=inputs,
            width=1000,
            **constants
        )
        
        trainer = pl.Trainer(
            logger=logger,
            **trainer_args
        )
        trainer.fit(model, train_loader, test_loader)
        return trainer.validate(model, test_loader)[-1]["test_loss_space"]

    def callback(study, trial):
        global model
        global best_model
        if study.best_trial.number != trial.number: return
        best_model = model

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5, n_warmup_steps=30, interval_steps=10
    )
    study = optuna.create_study(
        direction="minimize",
        pruner=pruner
    )

    study.optimize(objective, n_trials=15, callbacks=[callback])
    multiplier = study.best_trial.params["space"]

    # ------------------------
    # Load & Update DataFrame
    # ------------------------
    params = vars(args)
    params["time"] = datetime.datetime.now()
    params["cspace_base"] = base
    params["cspace_mul"] = multiplier
    params["cspace"] = base * multiplier
    df = pd.DataFrame([params])
    try:
        prev_df = pd.read_pickle("hyperparameters.pkl")
        df = pd.concat([prev_df, df], ignore_index=True)
    except FileNotFoundError:
        pass

    # ------------------------
    # Create & Train Optimal Model
    # ------------------------
    logger = pl.loggers.CSVLogger("", "logging", df.index.max())
    constants["cspace"] = base * multiplier
    model = utils.TYPES[args.type](
        utils.MODELS[args.model],
        inputs=inputs,
        width=1000,
        **constants
    )
    trainer = pl.Trainer(logger=logger, **utils.TRAINER_ARGS)
    trainer.fit(model, train_loader, test_loader)

    # ------------------------
    # Save Model & Parameters
    # ------------------------
    df.to_pickle("hyperparameters.pkl")
    trainer.save_checkpoint(os.path.join("checkpoints", f"{df.index.max()}.ckpt"))

    print("=========\nFinished!\n=========")
