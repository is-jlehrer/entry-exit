import os
import pathlib
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import argparse

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
from torchvision import transforms
from torchvision.models import ConvNeXt_Large_Weights, ResNet18_Weights

here = pathlib.Path(__file__).parent.resolve()
from utils import DECOMP_PATH


def generate_dataloaders():
    transform = {
        "train": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }

    train = StandardImageDataset(
        root=os.path.join(DECOMP_PATH, "train"), transform=transform["train"], label_map={"inside": 1, "outside": 0}
    )
    val = StandardImageDataset(
        root=os.path.join(DECOMP_PATH, "val"), transform=transform["val"], label_map={"inside": 1, "outside": 0}
    )

    train = DataLoader(train, shuffle=True, batch_size=64, num_workers=32)
    val = DataLoader(val, shuffle=False, batch_size=64, num_workers=32)

    return train, val

def calculate_weights():
    n_inside = len(os.listdir(os.path.join(DECOMP_PATH, "train", "inside")))
    n_outside = len(os.listdir(os.path.join(DECOMP_PATH, "train", "outside")))
    s = n_inside + n_outside

    return torch.from_numpy(np.array([s / (2 * n_outside), s / (2 * n_inside)]))


def generate_parser():
    parser = argparse.ArgumentParser(
        usage="""Train a model with the given batch size, wandb run name, and torchvision class. 
        Can specify weight decay, lr, and momentum as well."""
    )

    parser.add_argument("--name", required=False, default="resnet18")

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
        default=1e-4,
        type=float,
    )

    parser.add_argument(
        '--class-weights',
        required=False,
        default=False,
        action='store_true'
    )

    args = vars(parser.parse_args())

    return args


if __name__ == "__main__":
    params = generate_parser()
    train, val = generate_dataloaders()

    model = eval(f"models.{params['model']}()")
    model.fc = nn.Linear(in_features=model.fc.in_features, out_features=2)
    print('Weights are', calculate_weights())

    os.makedirs(os.path.join(here, params["name"]), exist_ok=True)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    optimizer = optim.SGD(
        params=model.parameters(),
        lr=float(params["lr"]),
        momentum=float(params["momentum"]),
        weight_decay=float(params["weight_decay"]),
    )
    train_handler = TrainModel(
        base_model=model,
        trainer_config={
            "max_epochs": 500,
            "logger": WandbLogger(project="Julian EntryExit", name=params["name"]),
            "callbacks": [
                ModelCheckpoint(
                    dirpath=os.path.join(here, params["name"]),
                    filename="model-{epoch}",
                    monitor="val_loss",
                    mode="min",
                    save_top_k=-1,
                ),
                LearningRateMonitor(logging_interval="epoch"),
                # EarlyStopping(monitor="val_loss", patience=5),
                StochasticWeightAveraging(swa_lrs=0.01),
            ],
            "track_grad_norm": 2,
            "accelerator": "gpu",
            "devices": 1,
        },
        model_config={
            "optimizer": optimizer,
            "scheduler": optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.75),
            "loss": nn.CrossEntropyLoss(weight=calculate_weights() if "class_weights" in params else None),
            "metrics": {
                "accuracy": Accuracy(average="macro", num_classes=2).to(device),
                "precision": Precision(average="macro", num_classes=2).to(device),
                "recall": Recall(average="macro", num_classes=2).to(device),
                "f1": F1Score(average="macro", num_classes=2).to(device),
            },
        },
    )

    train_handler.fit(train, val)
