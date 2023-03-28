import struct
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl



class PositiveLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ln_weights = nn.Parameter(torch.Tensor(out_features, in_features))
        self.biases = nn.Parameter(torch.Tensor(out_features))
        nn.init.xavier_normal_(self.ln_weights) # Random init
        #nn.init.xavier_normal_(self.biases) # Random init

    def forward(self, x):
        return F.linear(x, self.ln_weights.exp(), self.biases)

class ConcaveReLU(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            return -F.relu_(-x)
        else:
            return -F.relu(-x)

class LitModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()

        # Create model
        self.layer1a = PositiveLinear(1, 32)
        self.layer2a = PositiveLinear(32, 32)
        self.layer3a = PositiveLinear(32, 1)
        self.relu = nn.ReLU(inplace=True)

        self.layer1b = PositiveLinear(1, 32)
        self.layer2b = PositiveLinear(32, 32)
        self.layer3b = PositiveLinear(32, 1)
        self.crelu = ConcaveReLU(inplace=True)

    def forward(self, x):
        # Convex half
        cx = self.layer1a(x)
        cx = self.relu(cx)
        cx = self.layer2a(cx)
        cx = self.relu(cx)
        cx = self.layer3a(cx).mean(dim=1, keepdim=True)

        # Concave half
        cv = self.layer1b(x)
        cv = self.relu(cv)
        cv = self.layer2b(cv)
        cv = self.relu(cv)
        cv = self.layer3b(cv).mean(dim=1, keepdim=True)

        return cx + cv

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss.item())
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), 0.1)



def loadDataset(filename, dtype=np.int32, itemcount=100_000):
    try:
        # Little endian
        # 8 bytes item count
        # Data... (U32, U64, F64)
        with open(filename, "rb") as f:
            contents = f.read()
            count = struct.unpack("<Q", contents[:8])[0]
            itemcount = min(count, itemcount)

            inps = np.zeros(itemcount, dtype=dtype)
            tgts = np.zeros_like(inps, dtype=np.int64)
            for i in range(itemcount):
                file_idx = i * (count // itemcount)
                if dtype == np.int32:
                    value = struct.unpack("<i", contents[8 + 4*file_idx:12 + 4*file_idx]) # HACK: using signed instead of unsigned for torch interop
                elif dtype == np.uint64:
                    value = struct.unpack("<Q", contents[8 + 8*file_idx:16 + 8*file_idx])
                elif dtype == np.float64:
                    value = struct.unpack("<d", contents[8 + 8*file_idx:16 + 8*file_idx])

                inps[i] = value[0]

            data = torch.utils.data.TensorDataset(torch.Tensor(inps[...,np.newaxis]), torch.Tensor(tgts[...,np.newaxis]))

            return True, data

    except FileNotFoundError:
        return False, None



if __name__ == "__main__":
    success, d = loadDataset("books_200M_uint32")
    if not success:
        print("Failed to load dataset")
        exit()

    loader = torch.utils.data.DataLoader(d, batch_size=64, shuffle=True, pin_memory=True)

    model = LitModel()
    trainer = pl.Trainer(accelerator="gpu", max_epochs=2)
    trainer.fit(model, loader)

    torch.onnx.export(model, torch.Tensor([[1]]), "model.onnx", verbose=True)
    input("Press any key to continue...")