import os
import pathlib
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from lightml.data.make_dataset import Loaders, StandardImageDataset
from lightml.models.train_model import TrainModel
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint, StochasticWeightAveraging
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, F1Score, Precision, Recall
import pytorch_lightning as pl
import os

here = pathlib.Path(__file__).parent.resolve()
from utils import get_transforms


def calculate_weights(path):
    n_inside = len(os.listdir(os.path.join(path, "train", "inside")))
    n_outside = len(os.listdir(os.path.join(path, "train", "outside")))
    s = n_inside + n_outside

    return torch.from_numpy(np.array([s / (2 * n_outside), s / (2 * n_inside)])).float()


def generate_dataloaders(path):
    transform = get_transforms()
    train = StandardImageDataset(
        root=os.path.join(path, "train"), transform=transform["train"], label_map={"inside": 1, "outside": 0}
    )
    print('Training set has path', os.path.join(path, "train"))
    val = StandardImageDataset(
        root=os.path.join(here, "..", "varying_vid_fixed_sample_decomp", "val", "val"),
        transform=transform["val"],
        label_map={"inside": 1, "outside": 0},
    )
    print('Validation set has path', os.path.join(here, "..", "varying_vid_fixed_sample_decomp", "val", "val"))

    train = DataLoader(train, shuffle=True, batch_size=64, num_workers=32)
    val = DataLoader(val, shuffle=True, batch_size=64, num_workers=32)

    return train, val


def generate_parser():
    parser = argparse.ArgumentParser(
        usage="""Train a model with the given batch size, wandb run name, and torchvision class. 
        Can specify weight decay, lr, and momentum as well."""
    )

    parser.add_argument("--name", required=True, type=str, help="Wandb run name and name of folder to save checkpoints to")

    parser.add_argument(
        "--model",
        required=False,
        default="resnet18",
    )

    parser.add_argument(
        "--batch-size",
        required=False,
        default=64,
        type=int,
    )

    parser.add_argument(
        "--lr",
        required=False,
        default=3e-4,
        type=float,
    )

    parser.add_argument(
        "--momentum",
        required=False,
        default=1e-4,
        type=float,
    )

    parser.add_argument(
        "--weight-decay",
        required=False,
        default=1e-5,
        type=float,
    )

    parser.add_argument(
        "--dataset-path",
        required=True,
        type=str,
    )

    parser.add_argument("--class-weights", required=False, default=False, action="store_true")

    args = vars(parser.parse_args())

    return args


def calculate_mean_std(loader, total=100):
    mean = 0.0
    std = 0.0
    for i, (images, _) in enumerate(tqdm(loader)):
        if i == total:
            break
        batch_samples = images.size(0)  # batch size (the last batch can have smaller size!)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)

    mean /= len(loader.dataset)
    std /= len(loader.dataset)
    return mean, std


class TorchModelCallback(pl.Callback):
    def __init__(self, path) -> None:
        super().__init__()
        os.makedirs(path, exist_ok=True)
        self.path = path

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        model = pl_module.model

        torch.save(model.state_dict(), os.path.join(self.path, f"model-checkpoint-epoch-{epoch}"))


if __name__ == "__main__":
    params = generate_parser()
    train, val = generate_dataloaders(params["dataset_path"])
    print("Dataset path is", params["dataset_path"])

    model = eval(f"models.{params['model']}()")
    model.fc = nn.Linear(in_features=model.fc.in_features, out_features=2)

    savingcallback = TorchModelCallback(path=os.path.join(here, f"torch_model_checkpoints_{params['name']}"))

    os.makedirs(os.path.join(here, params["name"]), exist_ok=True)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    optimizer = optim.Adam(
        params=model.parameters(),
        lr=float(params["lr"]),
        # momentum=float(params["momentum"]),
        weight_decay=float(params["weight_decay"]),
    )
    train_handler = TrainModel(
        base_model=model,
        trainer_config={
            "max_epochs": 500,
            "logger": WandbLogger(project="Julian EntryExit", name=params["name"]),
            "callbacks": [
                LearningRateMonitor(logging_interval="epoch"),
                StochasticWeightAveraging(swa_lrs=0.01),
                savingcallback,
            ],
            "accelerator": "gpu",
            "devices": 1,
            "limit_val_batches": 10,
        },
        model_config={
            "optimizer": optimizer,
            "scheduler": optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.75),
            "loss": nn.CrossEntropyLoss(weight=calculate_weights(params["dataset_path"]) if params["class_weights"] else None),
            "metrics": {
                "micro_accuracy": Accuracy().to(device),
                "precision": Precision(average="macro", num_classes=2).to(device),
                "recall": Recall(average="macro", num_classes=2).to(device),
                "f1": F1Score(average="macro", num_classes=2).to(device),
            },
        },
    )

    train_handler.fit(train, val)
