import os
import pathlib

import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from lightml.data.make_dataset import Loaders, StandardImageDataset
from lightml.models.train_model import TrainModel
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint, StochasticWeightAveraging
from pytorch_lightning.loggers import WandbLogger
from torchmetrics import Accuracy, F1Score, Precision, Recall
from torchvision import transforms
from torchvision.models import ResNet18_Weights, ConvNeXt_Large_Weights
from torch.utils.data import DataLoader
import argparse

here = pathlib.Path(__file__).parent.parent.resolve()
DECOMP_PATH = os.path.join(here, "data", "decomp_100_frames_balanced")


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
    train = StandardImageDataset(root=os.path.join(DECOMP_PATH, "train"), transform=transform["train"],)

    val = StandardImageDataset(root=os.path.join(DECOMP_PATH, "val"), transform=transform["val"],)

    train = DataLoader(train, shuffle=True, batch_size=64, num_workers=32)
    val = DataLoader(val, shuffle=False, batch_size=64, num_workers=32)

    return train, val


def generate_parser():
    parser = argparse.ArgumentParser(
        usage="""Train a model with the given batch size, wandb run name, and torchvision class. 
        Can specify weight decay, lr, and momentum as well."""
    )

    parser.add_argument("--name", required=False, default="resnet18")

    parser.add_argument(
        "--model", required=False, default="resnet18",
    )

    parser.add_argument(
        "--batch-size", required=False, default=64, type=int,
    )

    parser.add_argument(
        "--lr", required=False, default=3e-4, type=float,
    )

    parser.add_argument(
        "--momentum", required=False, default=1e-4, type=float,
    )

    parser.add_argument(
        "--weight-decay", required=False, default=1e-4, type=float,
    )

    args = vars(parser.parse_args())

    return args


if __name__ == "__main__":
    params = generate_parser()
    train, val = generate_dataloaders()

    model = eval(f"models.{params['model']}()")
    model.fc = nn.Linear(in_features=model.fc.in_features, out_features=2)

    os.makedirs(os.path.join(here, params["name"]), exist_ok=True)

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
                    filename=f"{params['name']}-" + "model-{epoch}",
                    monitor="val_loss",
                    mode="min",
                    save_top_k=-1,
                ),
                LearningRateMonitor(logging_interval="epoch"),
                # EarlyStopping(monitor="val_loss", patience=5),
                StochasticWeightAveraging(swa_lrs=0.01),
            ],
            "log_every_n_steps": 20,
            "track_grad_norm": 2
        },
        model_config={
            "optimizer": optimizer,
            "scheduler": optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.75),
            "loss": nn.CrossEntropyLoss(),
            "metrics": {
                "accuracy": Accuracy(average="macro", num_classes=2),
                "precision": Precision(average="macro", num_classes=2),
                "recall": Recall(average="macro", num_classes=2),
                "f1": F1Score(average="macro", num_classes=2),
            },
        },
    )

    train_handler.fit(train, val)
