import os
import sys

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
        print(active_preds, times)
        active_preds = torch.stack([x[1] for x in active_preds])
        active_preds = F.softmax(active_preds)
        active_preds = active_preds.numpy()
        active_preds = self.moving_average(active_preds, n=2)
        active_preds = self.threshold(active_preds)

        return active_preds
        

if __name__ == "__main__":
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, 2)

    inference_wrapper = EntryExitInference(
        base_model=model,
        weights_path=os.path.join(here, 'default_checkpoints/model-epoch=10.ckpt'),
    )

    holdout_csv = format_data_csv(os.path.join(here, '..', 'data', 'holdout_data.csv'))

    preds = inference_wrapper.predict_from_uris(
        uri_list=holdout_csv["origin_uri"].values[0:1],
        local_path=os.path.join(here, '..', 'data', 'holdout'),
        sample_rate=20,  # predict even 10 frames 
        batch_size=64,
    )

    print(preds)
