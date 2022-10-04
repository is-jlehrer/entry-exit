import os
import pathlib

import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from lightml.data.make_dataset import Loaders
from lightml.models.train_model import TrainModel
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint,
                                         StochasticWeightAveraging)
from pytorch_lightning.loggers import WandbLogger
from torchmetrics import Accuracy, F1Score, Precision, Recall
from torchvision import transforms
from torchvision.models import ResNet18_Weights, ConvNeXt_Large_Weights

base = pathlib.Path(__file__).parent.parent.resolve()
DECOMP_PATH = os.path.join(base, "data", "decomp_100_frames_balanced")

if __name__ == "__main__":
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
    loader_gen = Loaders(
        root=DECOMP_PATH, 
        train_transform=transform["train"], 
        val_transform=transform["val"],
        batch_size=256,
        num_workers=64,
        shuffle=True,
    )

    train, val = loader_gen.get_loaders()

    # Define our model and make the linear layer have 2 ouptuts
    # instead of the 1000 by default
    model = models.resnet18()
    model.fc = nn.Linear(in_features=model.fc.in_features, out_features=2)

    optimizer = optim.SGD(params=model.parameters(), lr=0.0001, momentum=0)
    train_handler = TrainModel(
        base_model=model,
        trainer_config={
            "max_epochs": 200,
            "logger": WandbLogger(project="Julian EntryExit", name="baseline - resnet50"),
            "callbacks": [
                ModelCheckpoint(
                    dirpath="./default_checkpoints/", filename="model-{epoch}", monitor="val_loss", mode="min", save_top_k=-1,
                ),
                LearningRateMonitor(logging_interval="epoch"),
                EarlyStopping(monitor="val_loss", patience=5),
                StochasticWeightAveraging(swa_lrs=0.1),
            ],
            "log_every_n_steps": 20,
        },
        model_config={
            "optimizer": optimizer,
            "scheduler": optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.75),
            "loss": nn.CrossEntropyLoss(),
            "metrics": {
                "accuracy": Accuracy(average='macro', num_classes=2),
                "precision": Precision(average='macro', num_classes=2),
                "recall": Recall(average='macro', num_classes=2),
                "f1": F1Score(average='macro', num_classes=2),
            },
        },
    )

    train_handler.fit(train, val)
