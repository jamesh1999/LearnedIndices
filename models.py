import torch
from torch import nn

import pytorch_lightning as pl



class SpaceConservingLoss(object):
    def __init__(self, p=2.0):
        self.p = p

    def __call__(self, a, b):
        # Summing dimensions
        sum_a = tuple(range(1, a.dim()))
        sum_b = tuple(range(1, b.dim()))

        # Pair up inputs / encodings
        pair_a = torch.chunk(a, 2)
        pair_b = torch.chunk(b, 2)

        # L2 distances
        da = pair_a[1] - pair_a[0]
        da = da.square().sum(sum_a).abs().sqrt()
        db = pair_b[1] - pair_b[0]
        db = db.square().sum(sum_b).abs().sqrt()

        return (da - db).abs().float_power(self.p).mean() # p-norm error



class BasicNN(nn.Module):
    def __init__(self, inputs=20, width=400, depth=3):
        super().__init__()

        layers = []
        if depth == 1:
            layers += [nn.Linear(inputs, 1)]
        else:
            layers += [nn.Linear(inputs, width), nn.ReLU(True)]
            for _ in range(depth - 2):
                layers += [nn.Linear(width, width), nn.ReLU(True)]
            layers += [nn.Linear(width, 1)]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class BasicConvAutoencoder(object):
    class Encoder(nn.Module):
        def __init__(self, dimensions=20):
            super().__init__()
            
            self.dimensions = dimensions
            self.cnn = nn.Sequential(
                nn.Conv2d(1, 8, 3, stride=2, padding=1),
                nn.ReLU(True),
                nn.Conv2d(8, 16, 3, stride=2, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(True),
                nn.Conv2d(16, 32, 3, stride=2, padding=0),
                nn.ReLU(True)
            )
            self.flatten = nn.Flatten(start_dim=1)
            self.fully = nn.Sequential(
                nn.Linear(3 * 3 * 32, 128),
                nn.ReLU(True),
                nn.Linear(128, dimensions)
            )
            
        def forward(self, x):
            x = self.cnn(x)
            x = self.flatten(x)
            x = self.fully(x)
            return x
    
    class Decoder(nn.Module):
        
        def __init__(self, dimensions=20):
            super().__init__()
            self.fully = nn.Sequential(
                nn.Linear(dimensions, 128),
                nn.ReLU(True),
                nn.Linear(128, 3 * 3 * 32),
                nn.ReLU(True)
            )
            self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, 3, 3))
            self.cnn = nn.Sequential(
                nn.ConvTranspose2d(32, 16, 3, 
                stride=2, output_padding=0),
                nn.BatchNorm2d(16),
                nn.ReLU(True),
                nn.ConvTranspose2d(16, 8, 3, stride=2, 
                padding=1, output_padding=1),
                nn.BatchNorm2d(8),
                nn.ReLU(True),
                nn.ConvTranspose2d(8, 1, 3, stride=2, 
                padding=1, output_padding=1)
            )
            
        def forward(self, x):
            x = self.fully(x)
            x = self.unflatten(x)
            x = self.cnn(x)
            return x

    encoder = Encoder
    decoder = Decoder
    params = {}



# ----------------
# Lightning models
# ----------------

class DimensionReductionModel(pl.LightningModule):
    def __init__(self, model, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.k_distances = 1

        self.encoder_loss = nn.BCEWithLogitsLoss()
        self.space_loss = SpaceConservingLoss()

        # Create model
        self.encoder = model.encoder(**model.params, **kwargs)
        self.decoder = model.decoder(**model.params, **kwargs)

        self.dimensions = self.encoder.dimensions

    def forward(self, x):
        enc = self.encoder(x)
        dec = self.decoder(enc)
        return (enc, dec)

    def loss(self, inp, enc, dec):
        eloss = self.encoder_loss(inp, dec) # Auto-encoder loss
        sloss = self.space_loss(inp, enc) # Space loss

        loss = (1 - self.k_distances) * eloss + self.k_distances * sloss
        return (eloss, sloss, loss)

    def training_step(self, batch, batch_idx):
        enc, dec = self(batch)
        eloss, sloss, loss = self.loss(batch, enc, dec)

        self.log("train_eloss", eloss.item())
        self.log("train_sloss", sloss.item())
        self.log("train_loss", loss.item())
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def validation_step(self, batch, batch_idx):
        enc, dec = self(batch)
        _, _, loss = self.loss(batch, enc, dec)

        self.log("test_loss", loss.item())
        return loss

    def predict_step(self, batch, batch_idx):
        enc, _ = self(batch)
        return enc

class SFCModel(pl.LightningModule):
    def __init__(self, model, p=2.0, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.loss = SpaceConservingLoss(p=p)
        self.model = model(**kwargs)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        out = self(batch)
        loss = self.loss(batch, out)
        self.log("train_loss", loss.item())
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        loss = self.loss(batch, out)
        self.log("test_loss", loss.item())
        return loss

class RegressionModel(pl.LightningModule):
    def __init__(self, model, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.loss = nn.MSELoss()
        self.model = model(**kwargs)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y, y_hat)
        self.log("train_loss", loss.item())
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y, y_hat)
        self.log("test_loss", loss.item())
        return loss
    
class SingleDimensionClassifierModel(pl.LightningModule):
    def __init__(self, model, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.loss = nn.MSELoss()
        self.model = model(**kwargs)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y, y_hat)
        self.log("train_loss", loss.item())
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y, y_hat)
        self.log("test_loss", loss.item())
        return loss
    
class BucketIndexModel(pl.LightningModule):
    def __init__(self, model, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.loss = nn.MSELoss()
        self.model = model(**kwargs)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y, y_hat)
        self.log("train_loss", loss.item())
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y, y_hat)
        self.log("test_loss", loss.item())
        return loss