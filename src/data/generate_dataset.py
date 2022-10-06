# Gross but just temporary
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pathlib
import tqdm
import boto3
import cv2 as cv
from PIL import Image
import concurrent.futures
import pandas as pd

from lightml.data.decomp import DecompFromDataFrame
from utils import format_data_csv, DECOMP_PATH

curr = pathlib.Path(__file__).parent.resolve()

client = boto3.client("s3")


def download_from_uri(uri, local_path):
    """Downloads a file from S3 onto the local machine, where the URI
    is structured like: https://s3.amazonaws.com/is-raw-cec/20200117/4.mp4

    :param uri:
    :param s3_client: s3 client object
    :param local_path: Path to local directory to download files to
    :return: The s3 filename
    """
    split = uri.split("/")
    filename = "/".join(split[4:])
    bucket_name = split[3]
    if not os.path.isfile(local_path):
        try:
            client.download_file(bucket_name, filename, local_path)
        except Exception as e:  # pylint: disable=bare-except
            print(e, "continuing...")

    return filename


def decomp_all_from_one_vid(vid_row, local_path, format):
    st, et, uri, local = (
        vid_row["start_time"],
        vid_row["end_time"],
        vid_row["origin_uri"],
        vid_row["local_path"],
    )

    print(f'Downloading {uri}')
    download_from_uri(uri, local)
    cap = cv.VideoCapture(local)

    outside_path = "outside"
    inside_path = "inside"
    os.makedirs(os.path.join(local_path, outside_path), exist_ok=True)
    os.makedirs(os.path.join(local_path, inside_path), exist_ok=True)
    success = cap.grab()  # get the first frame
    frame_number = 1  # Don't want first modulus check to be True
    total_saved = 0

    print('Performing decomp')
    while success:
        if frame_number % 10 == 0:
            time = cap.get(cv.CAP_PROP_POS_MSEC)
            try:
                success, img = cap.retrieve()
            except Exception as e:
                print("Error when trying to recieve frame")
                print(e)
            
            # Outside
            if time < st or time > et:
                impath = os.path.join(
                    local_path,
                    outside_path,
                    f"{frame_number}{format}",
                )
            else:  # inside
                impath = os.path.join(
                    local_path,
                    inside_path,
                    f"{frame_number}{format}",
                )

            #  CV2 saves to BGR, PIL.Image uses RGB so we need to convert
            #  Since we train using PIL.Image.open().ToTensor()
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img.save(impath)
            total_saved += 1
        else:
            success = cap.grab()

        frame_number += 1

    print('Done, deleting video')
    os.remove(local)


def decomp_all_files(files, n_workers, local_path, format):
    os.makedirs(local_path, exist_ok=True)
    with tqdm.tqdm(total=len(files)) as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = []
            for i in range(len(files)):
                # More inefficient than apply but still < 200 vids generally. Slowdown is marginal
                futures.append(
                    executor.submit(
                        decomp_all_from_one_vid,
                        files.loc[i, :],
                        local_path,
                        format,
                    )
                )
            for future in concurrent.futures.as_completed(futures):
                pbar.update(1)
                if future.exception():
                    print(f"Exception at {future}, {future.exception()}")


if __name__ == "__main__":
    # perform decomp for each video
    # handler = DecompFromDataFrame(
    #     files=format_data_csv(os.path.join(curr, "combined_data.csv")),
    #     frames_per_clip=500,
    #     frames_per_nonclip=500,
    #     local_path=os.path.join(DECOMP_PATH),
    #     img_format=".jpg",
    #     val_prop=0.2,
    #     test_prop=0.05,
    #     max_workers=8,
    # )

    # handler.decomp()

    print('Reading in csv...')
    train = format_data_csv(os.path.join(curr, "train_na_stratified.csv"))
    val = format_data_csv(os.path.join(curr, "val_na_stratified.csv"))
    test = format_data_csv(os.path.join(curr, "test_na_stratified.csv"))

    decomp_all_files(train, local_path=os.path.join(curr, '..', 'full_decomp', 'train'), n_workers=8, format=".png")
    decomp_all_files(val, local_path=os.path.join(curr, '..', 'full_decomp', 'val'), n_workers=8, format=".png")
    decomp_all_files(test, local_path=os.path.join(curr, '..', 'full_decomp', 'test'), n_workers=8, format=".png")
