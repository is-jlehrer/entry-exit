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
from utils import format_data_csv, get_transforms
import pandas as pd
import cv2 as cv
import typing as tp
from PIL import Image
import tqdm
from lightml.common.defaults import download_from_uri

device = "cuda" if torch.cuda.is_available() else "cpu"


class InferenceModel:
    def __init__(
        self,
        base_model,
        weights_path=None,
        transform=None,
    ) -> None:
        self.transform = transform
        self.weights_path = weights_path
        self.model = base_model

        if weights_path is None:
            self.model = base_model
        else:
            try:
                self.model.load_state_dict(torch.load(weights_path))
            except Exception as e:
                print(f"Couldn't load {weights_path} as PyTorch model, trying PyTorch Lightning")
                print(e)
        print(f"Moving model to device: {device}")
        self.model.to(device)
        self.model.eval()

    def predict_from_uris(self, uris, local_path, postprocess=True, **kwargs):
        os.makedirs(local_path, exist_ok=True)
        preds = []
        for idx in uris.index:
            uri, st, et = uris.loc[idx, 'origin_uri'], uris.loc[idx, 'start_time'], uris.loc[idx, 'end_time']
            local_file = os.path.join(local_path, uri.split("/")[-1])
            download_from_uri(uri, local_file)

            out = self._predict_video(local_path=local_file, start_time=st, end_time=et, **kwargs)
            preds.append(self.postprocess(out) if postprocess else out)

            os.remove(local_file)

        return preds

    def _predict_video(
        self,
        local_path: str,
        start_time: int,
        end_time: int,
        sample_rate: int = 1,
        batch_size: int = 4,
        start_frame: int = 0,
        end_frame: int = None,
    ):
        """Calculates forward pass across a video file using the pretrained base model

        :param local_path: path to video to do inference on
        :param sample_rate: Predict every sample_rate frame, defaults to 1 (every frame)
        :param batch_size: Batch size for forward pass in model, defaults to 4
        :param start_frame: Frame to start inference on, defaults to 0 (first frame)
        :param end_frame: Frame to end inference on, defaults to None (last frame)
        :return: List containing model outputs for each frame
        """
        cap = cv.VideoCapture(local_path)
        success = cap.grab()  # get the next frame
        fno, total_batched = 0, 0

        preds: tp.List[torch.Tensor] = []
        batch: tp.List[torch.Tensor] = []
        times_milliseconds: tp.List[int] = []
        temp_imgs = []

        total_frames = cap.get(cv.CAP_PROP_FRAME_COUNT) / sample_rate if end_frame is None else end_frame / sample_rate

        with tqdm.tqdm(total=int(total_frames)) as pbar:
            while success:
                time = cap.get(cv.CAP_PROP_POS_MSEC)                
                # Skip frames until we need to do inference
                if start_frame > 0 and fno < start_frame:
                    success = cap.read()
                    fno += 1
                    continue

                # if we've hit the end frame drop out of this calcuation
                if end_frame is not None and fno >= end_frame:
                    break

                if fno % sample_rate == 0:
                    _, img = cap.retrieve()
                    temp_imgs.append(img)

                    # same as in decomp!
                    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                    img = Image.fromarray(img)
                    img = self.transform(img)

                    # collect image and frame index (to get time)
                    batch.append(img)
                    times_milliseconds.append(cap.get(cv.CAP_PROP_POS_MSEC))
                    pbar.update(1)

                if len(batch) == batch_size or end_frame is not None and fno >= end_frame:
                    total_batched += batch_size
                    with torch.no_grad():
                        batch = torch.stack(batch).to(device)
                        out = self.model(batch)

                    preds.extend(out)
                    batch = []
                    temp_imgs = []

                # outside
                if time < start_time or time > end_time:
                    maxs = F.softmax(out, dim=-1)
                    maxs = [x[1] for x in maxs]
                    false_positives_indices = (maxs > 0.5).nonzero().numpy().flatten()
                    for idx in false_positives_indices:
                        img = temp_imgs[idx]
                        prob = maxs[idx].item()
                        cv.imwrite(f"false_positive_{fno}_prob_{prob}.png", img)
                else:
                    # inside
                    maxs = F.softmax(out, dim=-1)
                    maxs = [x[1] for x in maxs]
                    false_negative_indices = (maxs < 0.3).nonzero().numpy().flatten()

                    for idx in false_negative_indices:
                        img = temp_imgs[idx]
                        prob = maxs[idx].item()
                        cv.imwrite(f"false_negative_{fno}_prob_{prob}.png", img)

                # reduce the batch size on the last batch to do the rest of the frames
                if 0 < int(total_frames - total_batched) < batch_size:
                    batch_size = int(total_frames - total_batched)
                    continue

                success = cap.grab()
                if not success:
                    if len(batch) > 0:
                        with torch.no_grad():
                            batch = torch.stack(batch).to(device)
                            out = self.model(batch)
                        preds.extend(out)
                fno += 1

        # Nicer to return it as a tensor for the end user
        return torch.stack(preds), times_milliseconds

    def postprocess(self, outputs, **kwargs):
        raise NotImplementedError(
            """User needs to implement postprocessing. Otherwise, run all prediction methods with postprocess=False"""
        )


here = pathlib.Path(__file__).parent.resolve()


def generate_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        help="path to model checkpoint",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--metadata",
        help="Path to csv for inference",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--name",
        help="File name suffix of saved probabilities + times for inference results (do not include extension)",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--limit",
        help="Number of videos to perform inference on",
        required=False,
        default=None,
        type=int,
    )

    return parser


class EntryExitInference(InferenceModel):
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
