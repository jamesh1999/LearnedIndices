import numpy as np

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
    
class TripletLoss(object):
    def __init__(self, distance_function=nn.PairwiseDistance(), **kwargs):
        self.distance = distance_function
        self.loss = nn.TripletMarginWithDistanceLoss(distance_function=distance_function, margin=10.0, **kwargs)

    def __call__(self, a, b):
        # a: batch, b: output
        # 0 - anchor, 1/2 - pos/neg

        # Distances
        da1 = self.distance(a[0], a[1])
        da2 = self.distance(a[0], a[2])

        # Find pos/neg datapoints
        mask = torch.gt(da2, da1).unsqueeze(1)
        pos = b[1]*mask    + b[2]*(~mask)
        neg = b[1]*(~mask) + b[2]*mask

        return self.loss(b[0], pos, neg)
    
class SpaceConservingLoss2(object):
    def __init__(self, p=2.0, repetitions=1, outer_dist=None, inner_dist=None):
        self.p = p
        self.repetitions = repetitions

        if inner_dist == None:
            inner_dist = outer_dist
        self.idist = inner_dist.torch
        self.odist = outer_dist.torch

    def __call__(self, a, b, device):

        # perm = torch.randperm(len(a), device=device)
        total_loss = None
        for i in range(1, self.repetitions+1):
            perm = torch.roll(torch.arange(0, len(a), device=device), i)

            # L2 distances
            da = self.odist(a, a[perm])
            db = self.idist(b, b[perm])

            if total_loss == None:
                total_loss = (da - db).abs_().mean() # p-norm error
            else:
                total_loss += (da - db).abs_().mean() # p-norm error

        return total_loss / self.repetitions
    
class TripletLoss2(object):
    def __init__(self, repetitions=1,
            outer_dist=None, inner_dist=None, dimensions=None, **kwargs):
        self.repetitions = repetitions
        self.dimensions = dimensions

        if inner_dist == None:
            inner_dist = outer_dist
        self.idist = inner_dist.torch
        self.odist = outer_dist.torch

        self.loss = nn.TripletMarginWithDistanceLoss(distance_function=self.idist, margin=0.1, **kwargs)

    def __call__(self, a, b, device):
        # a: batch, b: output
        # original: anchors, perm1/2: pos/neg

        total_loss = None
        for i in range(1, self.repetitions+1):
            perm1 = torch.roll(torch.arange(0, len(a), device=device), 3*i)
            perm2 = torch.roll(torch.arange(0, len(a), device=device), 5*i)

            # Distances
            da1 = self.odist(a, a[perm1])
            da2 = self.odist(a, a[perm2])

            # Find pos/neg datapoints
            mask = torch.gt(da2, da1).unsqueeze(1)

            pos = b[perm1]*mask    + b[perm2]*(~mask)
            neg = b[perm1]*(~mask) + b[perm2]*mask

            if total_loss == None:
                total_loss = self.loss(b, pos, neg)
            else:
                total_loss += self.loss(b, pos, neg)

        return total_loss / self.repetitions

class WeightedSpaceConservingLoss(object):
    def __init__(self, p=1.0, repetitions=1):
        self.p = p
        self.repetitions = repetitions

    def __call__(self, a, b, device):
        # Summing dimensions
        sum_a = tuple(range(1, a.dim()))
        sum_b = tuple(range(1, b.dim()))

        # perm = torch.randperm(len(a), device=device)
        total_loss = None
        for i in range(1, self.repetitions+1):
            perm = torch.roll(torch.arange(0, len(a), device=device), i)

            # L2 distances
            da = a - a[perm]
            da = da.square_().sum(sum_a).abs_().sqrt_()
            db = b - b[perm]
            db = db.square_().sum(sum_b).abs_().sqrt_()

            # Reciprocal of distance in input space: if this pair is close, it is important to maintain that in the output space
            # Reciprocal of distance in output space: if this pair is mapped close together, it had better be close in the input space
            # Smoothed
            weights = 1/da + 1/db

            if total_loss == None:
                total_loss = (weights * (da - db).abs_()).mean() # p-norm error
            else:
                total_loss += (weights * (da - db).abs_()).mean() # p-norm error

        return total_loss / self.repetitions

class WeightedSpaceConservingLoss2(object):
    def __init__(self, p=1.0, repetitions=1, outer_dist=None, inner_dist=None):
        self.p = p
        self.repetitions = repetitions

        if inner_dist == None:
            inner_dist = outer_dist
        self.idist = inner_dist.torch
        self.odist = outer_dist.torch

    def __call__(self, a, b, device):

        # perm = torch.randperm(len(a), device=device)
        total_loss = None
        for i in range(1, self.repetitions+1):
            perm = torch.roll(torch.arange(0, len(a), device=device), i)

            # L2 distances
            da = self.odist(a, a[perm])
            db = self.idist(b, b[perm])

            # Reciprocal of distance in input space: if this pair is close, it is important to maintain that in the output space
            # Reciprocal of distance in output space: if this pair is mapped close together, it had better be close in the input space
            # Smoothed
            weights = 1/(1+da) + 1/(1+db)

            if total_loss == None:
                total_loss = (weights * (da - db).abs_()).mean() # p-norm error
            else:
                total_loss += (weights * (da - db).abs_()).mean() # p-norm error

        return total_loss / self.repetitions
    
class MaximumMeanDiscrepancyLoss(object):
    def kernel(self, x, y): # Gaussian kernel, perhaps only suited to L2 distance?
        x_size = x.shape[0]
        y_size = y.shape[0]
        dim = x.shape[1]

        tiled_x = x.view(x_size,1,dim).repeat(1, y_size,1)
        tiled_y = y.view(1,y_size,dim).repeat(x_size, 1,1)

        return torch.exp(-torch.mean((tiled_x - tiled_y)**2,dim=2)/dim*1.0)

    def __call__(self, x, y):
        kxx = self.kernel(x, x)
        kyy = self.kernel(y, y)
        kxy = self.kernel(x, y)
        return torch.mean(kxx) - 2*torch.mean(kxy) + torch.mean(kyy)

class AveDistLoss(object):
    def __init__(self, dist):
        self.dist = dist.torch

    def __call__(self, a, b):
        return self.dist(a, b).abs_().mean()


class BasicNN(object):
    class Encoder(nn.Module):
        def __init__(self, inputs=-1, width=-1, depth=3, dimensions=20, vae=False, **kwargs):
            super().__init__()
            self.vae = vae

            if width == -1: width = inputs

            # Common part of VAE
            self.model = None
            if depth > 1:
                layers = [nn.Linear(inputs, width), nn.ReLU(True)]
                for _ in range(depth - 2):
                    layers += [nn.Linear(width, width), nn.ReLU(True)]
                self.model = nn.Sequential(*layers)

            self.mu = nn.Linear(inputs if depth == 1 else width, dimensions)
            if self.vae:
                self.var = nn.Linear(inputs if depth == 1 else width, dimensions)

        def forward(self, x):
            if self.model != None:
                x = self.model(x)
            mu = self.mu(x)
            if not self.vae:
                return mu
            var = self.var(x)
            return mu, var

    class Decoder(nn.Module):
        def __init__(self, inputs=-1, width=-1, depth=3, dimensions=20, **kwargs):
            super().__init__()

            if width == -1: width = inputs

            layers = []
            if depth == 1:
                layers += [nn.Linear(dimensions, inputs)]
            else:
                layers += [nn.Linear(dimensions, width), nn.ReLU(True)]
                for _ in range(depth - 2):
                    layers += [nn.Linear(width, width), nn.ReLU(True)]
                layers += [nn.Linear(width, inputs)]

            self.model = nn.Sequential(*layers)

        def forward(self, x):
            return self.model(x)
        
    encoder = Encoder
    decoder = Decoder
    params = {}



class BasicConv(object):
    class Encoder(nn.Module):
        def __init__(self, dimensions=20, vae=False, **kwargs):
            super().__init__()
            self.vae = vae

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
                nn.ReLU(True)
            )
            self.mu = nn.Linear(128, dimensions)
            if self.vae:
                self.var = nn.Linear(128, dimensions)
            
        def forward(self, x):
            x = self.cnn(x)
            x = self.flatten(x)
            x = self.fully(x)
            mu = self.mu(x)
            if not self.vae:
                return mu
            var = self.var(x)
            return mu, var
    
    class Decoder(nn.Module):
        
        def __init__(self, dimensions=20, **kwargs):
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
    
class ReductionVAE(pl.LightningModule):
    def __init__(self, model, calpha=1, clambda=3000, dimensions=20, outer_dist=None, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.ca = calpha
        self.cl = clambda
        self.dimensions = dimensions
        self.recon_loss = AveDistLoss(outer_dist)
        self.mmd_loss = MaximumMeanDiscrepancyLoss()

        # Create model
        self.encoder = model.encoder(vae=True, **model.params, dimensions=dimensions, **kwargs)
        self.decoder = model.decoder(**model.params, dimensions=dimensions, **kwargs)

    def forward(self, x):
        mu, lnvar = self.encoder(x)
        std = torch.exp(0.5 * lnvar)
        sample = mu + torch.randn_like(std) * std
        dec = self.decoder(sample)
        return mu, dec, lnvar
    
    def loss(self, inp, dec, mu, lnvar):
        samples = torch.randn((len(inp),self.dimensions), device=self.device)

        loss_r = self.recon_loss(inp, dec)
        loss_kl = -0.5 * torch.sum(1 + lnvar - mu*mu - lnvar.exp()) / inp.size(dim=0) # Mean. across batch
        loss_mmd = self.mmd_loss(samples, mu)
        loss_infovae = \
            loss_r + \
            (1-self.ca)*loss_kl + \
            (self.ca + self.cl - 1)*loss_mmd
        return loss_infovae, loss_r, loss_kl, loss_mmd

    def training_step(self, batch, batch_idx):
        mu, dec, lnvar = self(batch)
        loss, loss_r, loss_kl, loss_mmd = self.loss(batch, dec, mu, lnvar)
        self.log("train_loss", loss.item())
        self.log("train_lossr", loss_r.item())
        self.log("train_losskl", loss_kl.item())
        self.log("train_lossmmd", loss_mmd.item())
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def validation_step(self, batch, batch_idx):
        mu, dec, lnvar = self(batch)
        loss, _,_,_ = self.loss(batch, dec, mu, lnvar)
        self.log("test_loss", loss.item())
        return loss

    def predict_step(self, batch, batch_idx):
        enc, _, _ = self(batch)
        return enc
    
class ReductionUniversal(pl.LightningModule):
    def __init__(self, model, crecon=1, calpha=1, clambda=3000, cortho=0, cspace=0, ctriplet=0,
                 triplet_repetitions=1, dimensions=20, encoder_wd=0, decoder_wd=0, optimiser="adam", outer_dist=None, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.crecon = crecon
        self.calpha = calpha
        self.clambda = clambda
        self.cortho = cortho
        self.cspace = cspace
        self.ctriplet = ctriplet
        self.encoder_wd = encoder_wd
        self.decoder_wd = decoder_wd
        self.dimensions = dimensions
        self.optimiser = optimiser

        self.recon_loss = AveDistLoss(outer_dist)
        self.mmd_loss = MaximumMeanDiscrepancyLoss()
        self.space_loss = SpaceConservingLoss2()
        self.triplet_loss = TripletLoss2(dimensions=self.dimensions, repetitions=triplet_repetitions)

        # Create model
        self.encoder = model.encoder(vae=True, **model.params, dimensions=dimensions, **kwargs)
        self.decoder = model.decoder(**model.params, dimensions=dimensions, **kwargs)

    def forward(self, x):
        mu, lnvar = self.encoder(x)
        std = torch.exp(0.5 * lnvar)
        sample = mu + torch.randn_like(std) * std
        dec = self.decoder(sample)
        return mu, dec, lnvar
    
    def loss(self, inp, dec, mu, lnvar):
        samples = torch.randn((len(inp),self.dimensions), device=self.device)

        loss_recon = self.recon_loss(inp, dec)
        loss_kl = -0.5 * torch.sum(1 + lnvar - mu*mu - lnvar.exp()) / inp.size(dim=0) # Mean. across batch
        loss_mmd = self.mmd_loss(samples, mu)
        loss_ortho = 0#torch.norm(torch.matmul(mu.T, mu) - torch.eye(self.dimensions, device=self.device))
        loss_space = self.space_loss(inp, mu, self.device)
        loss_triplet = self.triplet_loss(inp, mu, self.device)
        loss = (
            self.crecon * loss_recon +
            (1-self.calpha)*loss_kl +
            (self.calpha + self.clambda - 1)*loss_mmd +
            self.cortho * loss_ortho +
            self.cspace * loss_space +
            self.ctriplet * loss_triplet
        )
        
        others = {
            "loss_recon":loss_recon,
            "loss_kl":loss_kl,
            "loss_mmd":loss_mmd,
            #"loss_ortho":loss_ortho,
            "loss_space":loss_space,
            "loss_triplet":loss_triplet
        }
        return loss, others

    def training_step(self, batch, batch_idx):
        mu, dec, lnvar = self(batch)
        loss, others = self.loss(batch, dec, mu, lnvar)
        self.log("train_loss", loss.item())
        for k, v in others.items():
            self.log(f"train_{k}", v.item())
        return loss

    def configure_optimizers(self):
        args = [
            {"params": self.encoder.parameters(), "weight_decay": self.encoder_wd},
            {"params": self.decoder.parameters(), "weight_decay": self.decoder_wd},
        ]

        if self.optimiser == "adam":
            return torch.optim.Adam(args)
        elif self.optimiser == "sgd":
            return torch.optim.SGD(args, 0.001)

    def validation_step(self, batch, batch_idx):
        mu, dec, lnvar = self(batch)
        loss, others = self.loss(batch, dec, mu, lnvar)
        self.log("test_loss", loss.item())
        for k, v in others.items():
            self.log(f"test_{k}", v.item())
        return loss

    def predict_step(self, batch, batch_idx):
        enc, _, _ = self(batch)
        return enc
    
class ReductionSCAE(pl.LightningModule):
    def __init__(self, model, crecon=0, cspace=0,
                 triplet_repetitions=1, dimensions=20, outer_dist=None, inner_dist=None, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.crecon = crecon
        self.cspace = cspace
        self.dimensions = dimensions

        self.recon_loss = AveDistLoss(outer_dist)
        self.space_loss = WeightedSpaceConservingLoss2(repetitions=triplet_repetitions, outer_dist=outer_dist, inner_dist=inner_dist)

        # Create model
        self.encoder = model.encoder(**model.params, dimensions=dimensions, **kwargs)
        self.decoder = model.decoder(**model.params, dimensions=dimensions, **kwargs)

    def forward(self, x):
        enc = self.encoder(x)
        dec = self.decoder(enc)
        return enc, dec
    
    def loss(self, inp, dec, enc):
        loss_recon = self.recon_loss(inp, dec)
        loss_space = self.space_loss(inp, enc, self.device)
        loss = self.crecon * loss_recon + self.cspace * loss_space
        
        others = {
            "loss_recon":loss_recon,
            "loss_space":loss_space
        }
        return loss, others

    def training_step(self, batch, batch_idx):
        enc, dec = self(batch)
        loss, others = self.loss(batch, dec, enc)
        self.log("train_loss", loss.item())
        for k, v in others.items():
            self.log(f"train_{k}", v.item())
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def validation_step(self, batch, batch_idx):
        enc, dec = self(batch)
        loss, others = self.loss(batch, dec, enc)
        self.log("test_loss", loss.item())
        for k, v in others.items():
            self.log(f"test_{k}", v.item())
        return loss

    def predict_step(self, batch, batch_idx):
        enc, _ = self(batch)
        return enc

class ReductionSpaceConserving(pl.LightningModule):
    def __init__(self, model, triplet_repetitions=1, power=2.0, outer_dist=None, inner_dist=None, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.loss = SpaceConservingLoss2(repetitions=triplet_repetitions, p=power, outer_dist=outer_dist, inner_dist=inner_dist)

        # Create model
        self.model = model.encoder(**model.params, **kwargs)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        out = self(batch)
        loss = self.loss(batch, out, self.device)
        self.log("train_loss", loss.item())
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        loss = self.loss(batch, out, self.device)
        self.log("test_loss", loss.item())
        return loss
    
class ReductionTriplet(pl.LightningModule):
    def __init__(self, model, triplet_repetitions=1, outer_dist=None, inner_dist=None, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.loss = TripletLoss2(repetitions=triplet_repetitions, outer_dist=outer_dist, inner_dist=inner_dist)

        # Create model
        self.model = model.encoder(**model.params, **kwargs)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        out = self(batch)
        loss = self.loss(batch, out, self.device)
        self.log("train_loss", loss.item())
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        loss = self.loss(batch, out, self.device)
        self.log("test_loss", loss.item())
        return loss
    
class ReductionWeightedSpaceConserving(pl.LightningModule):
    def __init__(self, model, triplet_repetitions=1, power=2.0, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.loss = WeightedSpaceConservingLoss(repetitions=triplet_repetitions, p=power)

        # Create model
        self.model = model.encoder(**model.params, **kwargs)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        out = self(batch)
        loss = self.loss(batch, out, self.device)
        self.log("train_loss", loss.item())
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        loss = self.loss(batch, out, self.device)
        self.log("test_loss", loss.item())
        return loss

class ReductionWeightedSpaceConserving2(pl.LightningModule):
    def __init__(self, model, triplet_repetitions=1, power=2.0, outer_dist=None, inner_dist=None, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.loss = WeightedSpaceConservingLoss2(repetitions=triplet_repetitions, p=power, outer_dist=outer_dist, inner_dist=inner_dist)

        # Create model
        self.model = model.encoder(**model.params, **kwargs)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        out = self(batch)
        loss = self.loss(batch, out, self.device)
        self.log("train_loss", loss.item())
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        loss = self.loss(batch, out, self.device)
        self.log("test_loss", loss.item())
        return loss

# Convenience to allow baseline tests to run with same workflow
class IdentityModel(pl.LightningModule):
    def __init__(self, model, **kwargs):
        super().__init__()
        self.save_hyperparameters()

    def forward(self, x):
        return x





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