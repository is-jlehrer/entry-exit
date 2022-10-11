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
from utils import format_data_csv, get_transforms
import pandas as pd

here = pathlib.Path(__file__).parent.resolve()
THRESH = 0.3

class EntryExitInference(InferenceModel):
    @staticmethod
    def moving_average(a, n=3):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

    def postprocess(self, outputs):
        # Instead of keeping logits [class_0_logit, class_1_logit], just take class 1
        active_preds, times = outputs
        
        active_preds = F.softmax(active_preds, dim=-1)
        active_preds = torch.stack([x[1] for x in active_preds])
        active_preds = active_preds.detach().cpu().numpy()
        
        return (active_preds, times)


if __name__ == "__main__":
    model = models.resnet50()
    model.fc = nn.Linear(model.fc.in_features, 2)

    inference_transform = get_transforms()["val"]
    inference_wrapper = EntryExitInference(
        base_model=model,
        weights_path=os.path.join(here, 'resnet50-longtrain/model-epoch=200.ckpt'),
        transform=inference_transform,
    )

    holdout_csv = format_data_csv(os.path.join(here, '..', 'data', 'val_na_stratified.csv'))
    uris = holdout_csv["origin_uri"].values

    preds = inference_wrapper.predict_from_uris(
        uri_list=uris,
        local_path=os.path.join(here, '..', 'data', 'holdout'),
        sample_rate=10,
        batch_size=64,
    )
    
    probas = pd.DataFrame([x[0] for x in preds])
    times = pd.DataFrame([x[1] for x in preds])

    probas.index = uris
    times.index = uris
    os.makedirs(os.path.join(here, 'inference'), exist_ok=True)

    probas.to_csv(os.path.join(here, 'inference/probs_validation_results_resnet50_longtrain.csv'))
    times.to_csv(os.path.join(here, 'inference/times_validation_results_resnet50_longtrain.csv'))
