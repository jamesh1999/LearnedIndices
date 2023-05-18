import ssl, logging, os, pickle, warnings

import h5py

import numpy as np
import pandas as pd

import torch
from torch import nn
from torchvision import transforms
from torchvision import datasets

from pytorch_lightning.core.module import LightningModule
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.callbacks.callback import Callback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers.logger import Logger

from optuna.integration import PyTorchLightningPruningCallback
import optuna
from optuna.storages._cached_storage import _CachedStorage
from optuna.storages._rdb.storage import RDBStorage

import models

DATASET_ARGS = {"root":".data", "download":True}

WORKER_COUNT = 0#4
BATCH_SIZE = 256
EPOCHS = 40

LOADER_ARGS = {"batch_size":BATCH_SIZE, "num_workers":WORKER_COUNT, "persistent_workers":WORKER_COUNT>0}

early_stop_callback = EarlyStopping(monitor="test_loss", patience=5, mode="min")
TRAINER_ARGS = {"accelerator":"gpu", "devices":"auto", "max_epochs":EPOCHS, "precision":16, "callbacks":[early_stop_callback], "gradient_clip_val":0.5, "detect_anomaly":False}

# Parameter values
DATASETS = {
    "fashionmnist":"fashion-mnist-784-euclidean.hdf5",
    "lastfm":"lastfm-64-dot.hdf5",
    "lasttrunc":"lasttrunc-64-angular.hdf5",
    "nytimes":"nytimes-256-angular.hdf5",
    "nytrunc":"nytrunc-256-angular.hdf5",
    "sift":"sift-128-euclidean.hdf5",
    "sifttrunc":"sifttrunc-128-euclidean.hdf5",
    "nytimesl2":"nytimes-256-euclidean.hdf5"
    }#{"fashionmnist":datasets.FashionMNIST}
MODELS = {"basicconv":models.BasicConv, "basicnn":models.BasicNN}
TYPES = {
        "basic":models.ReductionSpaceConserving,
        "triplet":models.ReductionTriplet,
        "relaxedold":models.ReductionWeightedSpaceConserving,
        "relaxed":models.ReductionWeightedSpaceConserving2,
        "vae":models.ReductionVAE,
        "identity":models.IdentityModel,
        "universal":models.ReductionUniversal,
        "scae":models.ReductionSCAE
    }
TYPE_TAGGED = {"autoenc":False, "basic":False, "relaxed":True}



_EPOCH_KEY = "ddp_pl:epoch"
_INTERMEDIATE_VALUE = "ddp_pl:intermediate_value"
_PRUNED_KEY = "ddp_pl:pruned"



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
            self.raw = np.copy(data).astype(np.float32)
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

class DFLogger(Logger):
    def __init__(self, version):
        self._version = version
        self.params = None
        self.metrics = pd.DataFrame()

    @property
    def name(self):
        return "DictionaryLogger"

    @property
    def version(self): return self._version

    def log_hyperparams(self, params):
        self.params = params

    def log_metrics(self, metrics, step):
        latest = pd.DataFrame(data=[metrics], index=[step])
        self.metrics = pd.concat([self.metrics, latest])

class PyTorchLightningPruningCallbackLatest(Callback):
    def __init__(self, trial: optuna.trial.Trial, monitor: str) -> None:
        super().__init__()

        self._trial = trial
        self.monitor = monitor
        self.is_ddp_backend = False

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.is_ddp_backend = trainer._accelerator_connector.is_distributed
        if self.is_ddp_backend:
            if version.parse(pl.__version__) < version.parse(  # type: ignore[attr-defined]
                "1.6.0"
            ):
                raise ValueError("PyTorch Lightning>=1.6.0 is required in DDP.")
            # If it were not for this block, fitting is started even if unsupported storage
            # is used. Note that the ValueError is transformed into ProcessRaisedException inside
            # torch.
            if not (
                isinstance(self._trial.study._storage, _CachedStorage)
                and isinstance(self._trial.study._storage._backend, RDBStorage)
            ):
                raise ValueError(
                    "optuna.integration.PyTorchLightningPruningCallback"
                    " supports only optuna.storages.RDBStorage in DDP."
                )
            # It is necessary to store intermediate values directly in the backend storage because
            # they are not properly propagated to main process due to cached storage.
            # TODO(Shinichi) Remove intermediate_values from system_attr after PR #4431 is merged.
            if trainer.is_global_zero:
                self._trial.storage.set_trial_system_attr(
                    self._trial._trial_id,
                    _INTERMEDIATE_VALUE,
                    dict(),
                )

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # Trainer calls `on_validation_end` for sanity check. Therefore, it is necessary to avoid
        # calling `trial.report` multiple times at epoch 0. For more details, see
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/1391.
        if trainer.sanity_checking:
            return

        current_score = trainer.callback_metrics.get(self.monitor)
        if current_score is None:
            message = (
                f"The metric '{self.monitor}' is not in the evaluation logs for pruning. "
                "Please make sure you set the correct metric name."
            )
            warnings.warn(message)
            return

        epoch = pl_module.current_epoch
        should_stop = False

        # Determine if the trial should be terminated in a single process.
        if not self.is_ddp_backend:
            self._trial.report(current_score.item(), step=epoch)
            if not self._trial.should_prune():
                return
            raise optuna.TrialPruned(f"Trial was pruned at epoch {epoch}.")

        # Determine if the trial should be terminated in a DDP.
        if trainer.is_global_zero:
            self._trial.report(current_score.item(), step=epoch)
            should_stop = self._trial.should_prune()

            # Update intermediate value in the storage.
            _trial_id = self._trial._trial_id
            _study = self._trial.study
            _trial_system_attrs = _study._storage.get_trial_system_attrs(_trial_id)
            intermediate_values = _trial_system_attrs.get(_INTERMEDIATE_VALUE)
            intermediate_values[epoch] = current_score.item()  # type: ignore[index]
            self._trial.storage.set_trial_system_attr(
                self._trial._trial_id, _INTERMEDIATE_VALUE, intermediate_values
            )

        # Terminate every process if any world process decides to stop.
        should_stop = trainer.strategy.broadcast(should_stop)
        trainer.should_stop = trainer.should_stop or should_stop
        if not should_stop:
            return

        if trainer.is_global_zero:
            # Update system_attr from global zero process.
            self._trial.storage.set_trial_system_attr(self._trial._trial_id, _PRUNED_KEY, True)
            self._trial.storage.set_trial_system_attr(self._trial._trial_id, _EPOCH_KEY, epoch)

class LpDist(object):
    def __init__(self, p=2.0):
        self._p = p
        self._pdist = nn.PairwiseDistance(p=p)
    def torch(self, a : torch.Tensor, b : torch.Tensor):
        return self._pdist(a, b)
    def numpy(self, a : np.ndarray, b : np.ndarray):
        return np.sum(abs(b-a) ** self._p, 1) ** (1/self._p)
    
class AngularDist(object):
    def __init__(self):
        self._cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    def torch(self, a : torch.Tensor, b : torch.Tensor):
        return self._cos(a, b)
    def numpy(self, a : np.ndarray, b : np.ndarray):
        norm_a = np.maximum(np.sqrt(np.sum(np.square(abs(a)), 1)), 1e-6)
        norm_b = np.maximum(np.sqrt(np.sum(np.square(abs(b)), 1)), 1e-6)
        return np.sum(norm_a * norm_b, 1)

FDISTANCES = {"euclidean":LpDist(p=2.0), "angular":AngularDist()}
DISTANCES = {
    "fashionmnist":FDISTANCES["euclidean"],
    "lastfm":FDISTANCES["angular"],
    "lasttrunc":FDISTANCES["angular"],
    "nytimes":FDISTANCES["angular"],
    "nytrunc":FDISTANCES["angular"],
    "sift":FDISTANCES["euclidean"],
    "sifttrunc":FDISTANCES["euclidean"],
    "nytimesl2":FDISTANCES["euclidean"]
    }



def init():
    # CIFAR10 has an expired cert
    ssl._create_default_https_context = ssl._create_unverified_context

    # Hide lightning warnings
    warnings.filterwarnings("ignore", ".*does not have many workers.*")
    logging.getLogger("lightning").setLevel(logging.ERROR)

    # Pandas display
    pd.set_option('display.max_rows', 500)

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