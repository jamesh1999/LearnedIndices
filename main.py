import torch
from torchvision import datasets

import pytorch_lightning as pl


import models, utils

DATASET = datasets.FashionMNIST



if __name__ == "__main__":

    # ------------------------
    # Prepare Datasets / Loaders
    # ------------------------

    train_data, test_data = utils.loadDataset(DATASET)
    train_loader   = utils.createDataLoader(train_data, shuffle=True)
    test_loader    = utils.createDataLoader(test_data)



    logger = pl.loggers.CSVLogger("")



    model = models.DimensionReductionModel(models.BasicConvAutoencoder, dimensions=24)
    trainer = pl.Trainer(logger=logger, **utils.TRAINER_ARGS)
    trainer.fit(model, train_loader, test_loader)
    trainer.save_checkpoint("reduce.ckpt")
