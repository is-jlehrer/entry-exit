import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import pathlib
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from lightml.models.predict_model import InferenceModel
from utils import format_data_csv, get_transforms
import pandas as pd

here = pathlib.Path(__file__).parent.resolve()
def generate_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--checkpoint',
        help='path to model checkpoint',
        required=True,
        type=str,
    )

    parser.add_argument(
        '--metadata',
        help='Path to csv for inference',
        required=True,
        type=str,
    )

    parser.add_argument(
        '--name',
        help='File name suffix of saved probabilities + times for inference results (do not include extension)',
        required=True,
        type=str,
    )

    parser.add_argument(
        '--limit',
        help='Number of videos to perform inference on',
        required=False,
        default=None,
        type=int,
    )

    return parser 

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

        print(f"{len(active_preds) = }, {len(times) = }")
        return (active_preds, times)


if __name__ == "__main__":
    parser = generate_parser()
    args = vars(parser.parse_args())

    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(args["checkpoint"]))
    
    inference_transform = get_transforms()["val"]
    inference_wrapper = EntryExitInference(
        base_model=model,
        transform=inference_transform,
    )

    holdout_csv = format_data_csv(args["metadata"], "", dropna=False)  # dont need path to decomped dataset, just leave blank
    uris = holdout_csv["origin_uri"].values if args["limit"] is None else holdout_csv["origin_uri"].values[0: args["limit"]]
    print("Doing inference on", len(uris), "number of videos")

    preds = inference_wrapper.predict_from_uris(
        uri_list=uris,
        local_path=os.path.join(here, "..", "data", "holdout"),
        sample_rate=10,  # predict every 50 frames
        batch_size=64,
    )

    probas = pd.DataFrame([x[0] for x in preds])
    times = pd.DataFrame([x[1] for x in preds])

    probas.index = uris
    times.index = uris

    tag = args["name"]
    os.makedirs(os.path.join(here, "inference"), exist_ok=True)
    probas.to_csv(os.path.join(here, f"inference/probs_{tag}.csv"))
    times.to_csv(os.path.join(here, f"inference/times_{tag}.csv"))
