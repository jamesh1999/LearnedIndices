import ssl, logging

import numpy as np

import torch
from torchvision import transforms

DATASET_ARGS = {"root":".data", "download":True}

WORKER_COUNT = 0#16
BATCH_SIZE = 256
EPOCHS = 40

LOADER_ARGS = {"batch_size":BATCH_SIZE, "num_workers":WORKER_COUNT, "persistent_workers":WORKER_COUNT>0}
TRAINER_ARGS = {"accelerator":"gpu", "devices":"auto", "max_epochs":EPOCHS, "precision":16}



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



def init():
    # CIFAR10 has an expired cert
    ssl._create_default_https_context = ssl._create_unverified_context

    # Hide lightning warnings
    logging.getLogger("lightning").setLevel(logging.ERROR)

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

def createDataLoader(dataset, **kwargs):
    return torch.utils.data.DataLoader(dataset, **kwargs, **LOADER_ARGS)

def numpyConvert(raw):
    data = raw.detach().cpu().numpy()
    data = data.reshape((data.shape[0], -1)) # Coallesce dimensions
    return data