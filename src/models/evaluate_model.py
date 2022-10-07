import os
import sys
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import pathlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import torchvision.models as models
from lightml.models.predict_model import InferenceModel
from utils import format_data_csv
import pandas as pd

here = pathlib.Path(__file__).parent.resolve()
THRESH = 0.3

class EntryExitInference(InferenceModel):
    @staticmethod
    def moving_average(a, n=3):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

    @staticmethod
    @np.vectorize
    def threshold(x):
        return 1 if x > THRESH else 0

    def postprocess(self, outputs):
        # Instead of keeping logits [class_0_logit, class_1_logit], just take class 0
        active_preds, times = outputs
        
        active_preds = torch.stack([x[1] for x in active_preds])
        active_preds = F.softmax(active_preds)
        active_preds = active_preds.numpy()
        # active_preds = self.moving_average(active_preds, n=15)
        # active_preds = self.threshold(active_preds)
        # active_preds = np.where(active_preds == 1)[0]  # list of indices where preds are 1 
        # return times[active_preds[0]], times[active_preds[-1]] if len(active_preds) >= 2 else np.nan, np.nan 
        return active_preds


if __name__ == "__main__":
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, 2)

    inference_wrapper = EntryExitInference(
        base_model=model,
        weights_path=os.path.join(here, 'resnet18-gpu-bigdecomp/model-epoch=19.ckpt'),
    )

    holdout_csv = format_data_csv(os.path.join(here, '..', 'data', 'test_na_stratified.csv'))

    preds = inference_wrapper.predict_from_uris(
        uri_list=holdout_csv["origin_uri"].values[0:2],
        local_path=os.path.join(here, '..', 'data', 'holdout'),
        sample_rate=10,  # predict every 50 frames
        batch_size=64,
        end_frame=100,
    )

    preds = pd.DataFrame(preds)
    preds.to_csv(os.path.join(here, 'model_results.csv'))


    # pred_entry = [x[0] for x in preds]
    # pred_exit = [x[1] for x in preds]

    # print(pred_entry, pred_exit)
    print(preds)
    os.makedirs("distplots", exist_ok=True)
    for i, pred in enumerate(preds):
        plt.scatter(list(range(len(preds[i]))), preds[i])
        plt.savefig(f"distplots/pred_distribution_vid_{i}.png")
        plt.clf()
