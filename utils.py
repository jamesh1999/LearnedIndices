import ssl, logging, os, pickle

import h5py

import numpy as np

import torch
from torchvision import transforms
from torchvision import datasets

from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import models

DATASET_ARGS = {"root":".data", "download":True}

WORKER_COUNT = 0#4
BATCH_SIZE = 128
EPOCHS = 40

LOADER_ARGS = {"batch_size":BATCH_SIZE, "num_workers":WORKER_COUNT, "persistent_workers":WORKER_COUNT>0}

early_stop_callback = EarlyStopping(monitor="test_loss", patience=8, mode="min")
TRAINER_ARGS = {"accelerator":"gpu", "devices":"auto", "max_epochs":EPOCHS, "precision":16, "callbacks":[early_stop_callback]}

# Parameter values
DATASETS = {"fashionmnist":"fashion-mnist-784-euclidean.hdf5", "lastfm":"lastfm-64-dot.hdf5", "nytimes":"nytimes-256-angular.hdf5", "sift":"sift-128-euclidean.hdf5"}#{"fashionmnist":datasets.FashionMNIST}
MODELS = {"basicconv":models.BasicConv, "basicnn":models.BasicNN}
TYPES = {
        "autoenc":models.ReductionAutoEncoder,
        "basic":models.ReductionSpaceConserving,
        "triplet":models.ReductionTriplet,
        "relaxed":models.ReductionWeightedSpaceConserving,
        "vae":models.ReductionVAE,
        "identity":models.IdentityModel
    }
TYPE_TAGGED = {"autoenc":False, "basic":False, "relaxed":True}



class TransformNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        return (sample - self.mean) / self.std

class TransformReshape(object):
    def __init__(self, shape):
        self.shape = shape

    def __call__(self, sample):
        return torch.reshape(sample, self.shape)



class UniformDataset(torch.utils.data.Dataset):
    def __init__(self, dimensions, length):
        self.data = torch.from_numpy(np.float32(np.random.uniform(0, 1, (length, dimensions))))
    def __getitem__(self, idx):
        return self.data[idx]
    def __len__(self):
        return len(self.data)

class UnlabeledDataset(torch.utils.data.Dataset):
    def __init__(self, original):
        self.data = torch.stack([inp for inp,_ in original])
    def __getitem__(self, idx):
        return self.data[idx]
    def __len__(self):
        return len(self.data)

class TaggedDataset(torch.utils.data.Dataset):
    def __init__(self, original):
        self.data = torch.stack([inp for inp in original])
    def __getitem__(self, idx):
        return self.data[idx], idx
    def __len__(self):
        return len(self.data)
    
class GroupedDataset(torch.utils.data.Dataset):
    def __init__(self, original, count=2):
        self.data = torch.stack([inp for inp in original])
        self.count=count
    def __getitem__(self, idx):
        others = []
        for i in range(self.count-1):
            others.append(self.data[np.random.randint(len(self.data))])
        return self.data[idx], *others
    def __len__(self):
        return len(self.data)
    
class H5Loader(object):
    class H5Dataset(torch.utils.data.Dataset):
        def __init__(self, data):
            self.raw = np.copy(data)
            self.t = lambda x: x
        def __getitem__(self, idx):
            return self.t(torch.from_numpy(self.raw[idx]))
        def __len__(self):
            return len(self.raw)
        
    def __init__(self, filename):
        with h5py.File(filename, 'r') as f:
            self.test = self.H5Dataset(f["test"])
            self.train = self.H5Dataset(f["train"])

            # Just load as numpy arrays
            self.neighbours = np.copy(f["neighbors"])
            self.distances = np.copy(f["distances"])

    def setTransform(self, transform):
        self.test.t = transform
        self.train.t = transform


def init():
    # CIFAR10 has an expired cert
    ssl._create_default_https_context = ssl._create_unverified_context

    # Hide lightning warnings
    logging.getLogger("lightning").setLevel(logging.ERROR)

def tryLoadPickle(filename, default):
    try:
        with open(filename, "rb") as pklfile:
            return pickle.load(pklfile)
    except FileNotFoundError:
        return default
    
def updatePickle(filename, data):
    with open(filename, "wb") as pklfile:
        pickle.dump(data, pklfile)

def loadDataset(dataset, transform=True, **kwargs):
    t = None
    if transform:
        t = [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]
        t = transforms.Compose(t)

    train = dataset(train=True, transform=t, **kwargs, **DATASET_ARGS)
    test = dataset(train=False, transform=t, **kwargs, **DATASET_ARGS)
    return UnlabeledDataset(train), UnlabeledDataset(test)

def loadH5Dataset(filename, transform=True, shape=(-1,)):
    path = os.path.join(".data2", filename)
    data = H5Loader(path)

    if not transform:
        return data
    
    # Avoid recomputing mean/std dev.
    cachefile = os.path.join(".data2", "cache.pkl")
    cache = tryLoadPickle(cachefile, {})
    if not filename in cache:
        cache[filename] = {
            "mean": np.mean(data.train.raw),
            "std": np.std(data.train.raw)
        }
        updatePickle(cachefile, cache)
    mean = cache[filename]["mean"]
    std = cache[filename]["std"]

    t = transforms.Compose([
        TransformNormalize(mean, std),
        TransformReshape(shape)
    ])
    data.setTransform(t)
    return data

def loadModel(param_df, idx):
    row = param_df.iloc[idx]
    path = os.path.join("checkpoints", f"{idx}.ckpt")
    return TYPES[row["type"]].load_from_checkpoint(
            path,
            model=MODELS[row["model"]]
        )

def createDataLoader(dataset, **kwargs):
    return torch.utils.data.DataLoader(dataset, **kwargs, **LOADER_ARGS)

def numpyConvert(raw):
    data = raw.detach().cpu().numpy()
    data = data.reshape((data.shape[0], -1)) # Coallesce dimensions
    return data