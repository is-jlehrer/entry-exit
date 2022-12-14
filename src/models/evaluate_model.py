import os
import sys
from xmlrpc.client import FastMarshaller
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
import seaborn as sns
from utils import format_data_csv, get_transforms
import pandas as pd
from sklearn.metrics import roc_curve, confusion_matrix, accuracy_score, f1_score, auc
from torchmetrics import Accuracy, F1Score

here = pathlib.Path(__file__).parent.resolve()
THRESH = 0.5


def generate_confusion_matrix(probs, times, truth):
    preds, truths = [], []

    for vid in probs.index:
        pred = probs.loc[vid, :]
        pred = pred[~np.isnan(pred)]
        pred = pred.apply(lambda x: int(x > THRESH))

        time = times.loc[vid, :]
        time = time[~np.isnan(time)].values

        st, et = truth.loc[vid, 'start_time'], truth.loc[vid, 'end_time']
        gt = [1 if t >= st and t <= et else 0 for t in time]
        if len(gt) > len(pred):
            print(f'WARNING: Missing {len(gt) - len(pred)} probabilities in confusion matrix calc. Continuing')
            gt = gt[0: len(pred)]

        preds.extend(pred)
        truths.extend(gt)

    normed_matrix = confusion_matrix(y_true=np.array(truths), y_pred=np.array(preds), normalize='all')
    matrix = confusion_matrix(y_true=np.array(truths), y_pred=np.array(preds))
    return normed_matrix, matrix

def generate_roc_curve(probs, times, truth):
    scores, truths = [], []

    for vid in probs.index:
        score = probs.loc[vid, :]
        score = score[~np.isnan(score)].values

        time = times.loc[vid, :]
        time = time[~np.isnan(time)].values

        st, et = truth.loc[vid, 'start_time'], truth.loc[vid, 'end_time']
        gt = [1 if t >= st and t <= et else 0 for t in time]
        if len(gt) > len(score):
            print(f'WARNING: Missing {len(gt) - len(score)} probabilities in roc curve calc. Continuing')
            gt = gt[0: len(score)]

        scores.extend(score)
        truths.extend(gt)
    
    curve = roc_curve(y_true=np.array(truths), y_score=np.array(scores))

    return curve

def calculate_validation_statistics(probs, times, truth):
    scores, truths = [], []
    for vid in probs.index:
        score = probs.loc[vid, :]
        score = score[~np.isnan(score)].values

        time = times.loc[vid, :]
        time = time[~np.isnan(time)].values

        st, et = truth.loc[vid, 'start_time'], truth.loc[vid, 'end_time']
        gt = [1 if t > st and t < et else 0 for t in time]
        score = [1 if x > 0.5 else 0 for x in score]

        if len(gt) != len(score):
            print(f'WARNING: Missing {len(time) - len(score)} probabilities in acc/f1 score. Continuing')
            print(f'len(scores)={len(score)} & len(times)={len(time)}')
            gt = gt[0: len(score)]

        scores.extend(score)
        truths.extend(gt)

    print('SKLEARN ACC IS', accuracy_score(truths, scores))
    print('SKLEARN F1 IS', f1_score(truths, scores))

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
        required=False,
        default="."
    )

    parser.add_argument(
        '--tag',
        help='Suffix of saved files',
        type=str,
        required=True,
    )

    return parser


if __name__ == "__main__":
    parser = generate_parser()
    args = vars(parser.parse_args())

    probs, times, truths, save, tag = args["probs"], args["times"], args["metadata"], args["save"], args["tag"]

    probs = pd.read_csv(probs, index_col='origin_uri')
    times = pd.read_csv(times, index_col='origin_uri')
    truths = format_data_csv(truths, '', dropna=True)  # decomp path doesnt matter, just leave blank
    truths.index = truths["origin_uri"]

    normed_matrix, matrix = generate_confusion_matrix(probs, times, truths)
    fpr, tpr, threshs = generate_roc_curve(probs, times, truths)
    results = calculate_validation_statistics(probs, times, truths)
    roc_auc = auc(fpr, tpr)
    print(f'AUC IS {roc_auc}')

    df_cm_norm = pd.DataFrame(normed_matrix, index=["outside", "inside"], columns=["outside", "inside"])
    df_cm = pd.DataFrame(matrix, index=["outside", "inside"], columns=["outside", "inside"])
    
    plt.figure(figsize=(10, 7))
    sns.heatmap(df_cm_norm, annot=True)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Population Normalized Confusion Matrix: Model V1")
    plt.savefig(f"confusion_matrix_norm_{tag}.png")

    plt.clf()
    plt.figure(figsize=(10, 7))
    sns.heatmap(df_cm, annot=True)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Confusion Matrix: Model V1")
    plt.savefig(f"confusion_matrix_{tag}.png")

    plt.clf()
    plt.figure(figsize=(10, 7))
    sns.lineplot(x=fpr, y=tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve: Model V1')
    plt.savefig(f"roc_curve_{tag}.png")

    plt.clf()
    plt.figure(figsize=(10, 7))
    sns.lineplot(x=np.log(fpr), y=np.log(tpr))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve: Model V1 (log scale)')
    plt.savefig(f"roc_curve_log_{tag}.png")
