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
from sklearn.metrics import roc_curve, confusion_matrix
from torchmetrics import Accuracy, F1Score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

here = pathlib.Path(__file__).parent.resolve()
THRESH = 0.5


def generate_confusion_matrix(probs, times, truth):
    preds, truths = [], []

    for vid in probs.index:
        pred = probs.loc[vid, :]
        pred = pred.apply(lambda x: int(x > THRESH))
        pred = pred[~np.isnan(pred)].values
        
        st, et = truth.loc[vid, 'start_time'], truth.loc[vid, 'end_time']
        gt = [1 if t >= st and t <= et else 0 for t in times.loc[vid, :].values]
        if len(gt) > len(pred):
            print('WARNING: Missing some probabilities. Continuing')
            gt = gt[0: len(pred)]

        preds.extend(pred)
        truths.extend(gt)

    matrix = confusion_matrix(y_true=np.array(truths), y_pred=np.array(preds), normalize='all')
    return matrix 

def generate_roc_curve(probs, times, truth):
    scores, truths = [], []
    for vid in probs.index:
        score = probs.loc[vid, :]
        score = score[~np.isnan(score)].values

        st, et = truth.loc[vid, 'start_time'], truth.loc[vid, 'end_time']
        gt = [1 if t >= st and t <= et else 0 for t in times.loc[vid, :].values]
        if len(gt) > len(score):
            print('WARNING: Missing some probabilities. Continuing')
            gt = gt[0: len(score)]

        scores.extend(score)
        truths.extend(gt)
    
    curve = roc_curve(y_true=np.array(truths), y_score=np.array(scores))

    return curve

def generate_validation_statistics(probs, times, truth):
    acc = Accuracy(multiclass=False, num_classes=1)
    f1 = F1Score(multiclass=False, num_classes=1)

    scores, truths = [], []
    for vid in truth.index:
        score = probs.loc[vid, :]
        score = score[~np.isnan(score)].values

        time = times.loc[vid, :]
        time = time[~np.isnan(time)].values

        st, et = truth.loc[vid, 'start_time'], truth.loc[vid, 'end_time']
        gt = [1 if t > st and t < et else 0 for t in time]
        score = [1 if x > 0.5 else 0 for x in score]

        if len(gt) != len(score):
            print('WARNING: Missing some probabilities. Continuing')
            print(f'len(scores)={len(score)} & len(times)={len(time)}')
            gt = gt[0: len(score)]

        scores.extend(score)
        truths.extend(gt)

    print(np.array(score), np.array(gt))
    print('SKLEARN ACC IS', accuracy_score(truths, scores))
    print('SKLEARN F1 IS', f1_score(truths, scores))
    return {
        "accuracy": acc(torch.tensor(scores), torch.tensor(truths)),
        "f1": f1(torch.tensor(scores), torch.tensor(truths)),
        "accdefault": Accuracy()(torch.tensor(scores), torch.tensor(truths))
    }

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

    probs = pd.read_csv(probs, index_col='Unnamed: 0')
    times = pd.read_csv(times, index_col='Unnamed: 0')
    truths = format_data_csv(truths, '', dropna=True)  # decomp path doesnt matter, just leave blank
    truths.index = truths["origin_uri"]

    # matrix_vals = generate_confusion_matrix(probs, times, truths)
    # fpr, tpr, threshs = generate_roc_curve(probs, times, truths)
    results = generate_validation_statistics(probs, times, truths)
    print(results)
    # df_cm = pd.DataFrame(matrix_vals, index=["outside", "inside"], columns=["outside", "inside"])
    
    # plt.figure(figsize=(10, 7))
    # sns.heatmap(df_cm, annot=True)
    # plt.xlabel("Actual")
    # plt.ylabel("Predicted")
    # plt.title("Population Normalized Confusion Matrix: Model V1")
    # plt.savefig(f"confusion_matrix_{tag}.png")

    # df_roc = pd.DataFrame({
    #     "True Positive Rate": fpr,
    #     "False Positive Rate": tpr,
    # })
    # plt.clf()
    # plt.figure(figsize=(10, 7))
    # sns.lineplot(x=fpr, y=tpr)
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('ROC Curve: Model V1')
    # plt.savefig(f"roc_curve_{tag}.png")