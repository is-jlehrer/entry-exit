import os
import sys
from tkinter import E
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import pathlib
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import torchvision.models as models
from lightml.models.predict_model import InferenceModel
from utils import format_data_csv, get_transforms
import pandas as pd
from sklearn.metrics import roc_curve, confusion_matrix

here = pathlib.Path(__file__).parent.resolve()
THRESH = 0.5


def generate_confusion_matrix(probs, times, truth):
    preds, truths = [], []

    probs = probs.apply(lambda x: int(x > THRESH))

    for vid in probs.index:
        pred = probs.loc[vid, :].values
        
        st, et = truth.loc[vid, 'start_time'], truth.loc[vid, 'end_time']
        gt = [1 if t >= st and t <= et else 0 for t in times.loc[vid, :].values]

        preds.extend(pred)
        truths.extend(gt)

    matrix = confusion_matrix(y_true=np.array(truths), y_pred=np.array(preds))
    return matrix 

def generate_roc_curve(probs, times, truth):
    scores, truths = [], []
    for vid in probs.index:
        score = probs.loc[vid, :].values
        st, et = truth.loc[vid, 'start_time'], truth.loc[vid, 'end_time']
        gt = [1 if t >= st and t <= et else 0 for t in times.loc[vid, :].values]

        scores.extend(score)
        truths.extend(gt)
    
    curve = roc_curve(y_true=np.array(truths), y_pred=np.array(scores))

    return curve

def generate_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--probs',
        help='Path to csv containing probabilities for each frame prediction',
        type=str,
        required=True
    )

    parser.add_argument(
        '--times',
        help='Path to file containing times for each frame prediction',
        type=str,
        required=True,
    )

    parser.add_argument(
        '--metadata',
        help='Path to file containing metadata (train, val, test split)',
        type=str,
        required=True,
    )

    parser.add_argument(
        '--save',
        help='Path to save roc/confusion matrices to',
        type=str,
        required=True,
    )

    return parser


if __name__ == "__main__":
    parser = generate_parser()
    args = vars(parser.parse_args())

    probs, times, truths, save = args["probs"], args["times"], args["metadata"], args["save"]

    probs = pd.read_csv(probs, index_col='Unnamed: 0')
    times = pd.read_csv(times, index_col='Unnamed: 0')
    truths = format_data_csv(truths, '')  # decomp path doesnt matter, just leave blank
    truths.index = truths["origin_uri"]

    matrix_vals = generate_confusion_matrix(probs, times, truths)
    roc_curve_vals = generate_roc_curve(probs, times, truths)


