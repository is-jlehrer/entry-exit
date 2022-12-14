# Gross but just temporary
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import concurrent.futures
import logging
import pathlib

import boto3
import cv2 as cv
import tqdm
from PIL import Image
from utils import format_data_csv

logging.basicConfig(
    filename="new_decomp_logfile",
    filemode="a",
    format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.DEBUG,
)
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


def decomp_all_from_one_vid(vid_row, local_path, format, outside_prop, inside_prop, tag):
    st, et, uri, local = (
        vid_row["start_time"],
        vid_row["end_time"],
        vid_row["origin_uri"],
        vid_row["local_path"],
    )

    print('Local is', local)

    print(f"Downloading {uri}")
    download_from_uri(uri, local)
    try:
        cap = cv.VideoCapture(local)
    except Exception as e:
        logging.exception(f"Error when doing cv.VideoCapture on {local}")

    outside_path = "outside"
    inside_path = "inside"
    os.makedirs(os.path.join(local_path, outside_path), exist_ok=True)
    os.makedirs(os.path.join(local_path, inside_path), exist_ok=True)

    success = cap.grab()  # get the first frame
    frame_number = 1
    in_procedure = False
    total_saved = 0

    outside_sample_rate = 1 // outside_prop
    inside_sample_rate = 1 // inside_prop

    print(f"Outside sample rate is {outside_sample_rate} and inside sample rate is {inside_sample_rate}")
    print("Performing decomp")
    while success:
        if frame_number % (outside_sample_rate if not in_procedure else inside_sample_rate) == 0:
            # print(f'Saving on frame {frame_number}')
            time = cap.get(cv.CAP_PROP_POS_MSEC)
            try:
                success, img = cap.retrieve()
            except Exception as e:
                print("Error when trying to recieve frame")
                print(e)

            # Outside
            if time < st or time > et:
                in_procedure = False
                impath = os.path.join(
                    local_path,

                    outside_path,
                    f"{tag}_{total_saved}_{frame_number}{format}",
                )
            else:  # inside
                in_procedure = True
                impath = os.path.join(
                    local_path,
                    inside_path,
                    f"{tag}_{total_saved}_{frame_number}{format}",
                )

            #  CV2 saves to BGR, PIL.Image uses RGB so we need to convert
            #  Since we train using PIL.Image.open().ToTensor()
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img.save(impath)
            total_saved += 1

        frame_number += 1
        success = cap.grab()

    print("Done, deleting video")
    os.remove(local)


def decomp_all_files(files, n_workers, local_path, format, outside_prop, inside_prop):
    os.makedirs(local_path, exist_ok=True)
    with tqdm.tqdm(total=len(files)) as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = []
            for row in files.iterrows():
                # More inefficient than apply but still < 200 vids generally. Slowdown is marginal
                futures.append(
                    executor.submit(
                        decomp_all_from_one_vid,
                        row[1],
                        local_path,
                        format,
                        outside_prop,
                        inside_prop,
                        row[0],
                    )
                )
            for future in concurrent.futures.as_completed(futures):
                pbar.update(1)
                if future.exception():
                    logging.exception(f"Exception at {future}, {future.exception()}")


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

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outside-prop",
        type=float,
        help="Proportion of frames to sample per video outside of the procedure",
        required=True,
    )

    parser.add_argument(
        "--inside-prop",
        type=float,
        help="Proportion of frames to sample per video inside of the procedure",
        required=True,
    )

    parser.add_argument(
        "--path",
        help="Path to do decomp to",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of processes to do decomp on",
        required=False,
    )

    parser.add_argument(
        "--num-vids", type=int, default=None, required=False, help="If passed, only decomp this number of training vids"
    )

    args = parser.parse_args()
    outside, inside, path, num_workers, num_vids = args.outside_prop, args.inside_prop, args.path, args.num_workers, args.num_vids
    print("Making decomp path")
    os.makedirs(path, exist_ok=True)

    train = format_data_csv(os.path.join(curr, "train_na_stratified.csv"), path)
    print(f'Total train vids is {len(train)}')

    if num_vids is not None:
        train = train.sample(num_vids)

    decomp_all_files(
        train,
        local_path=os.path.join(path, "train"),
        n_workers=num_workers,
        format=".png",
        outside_prop=outside,
        inside_prop=inside,
    )
    # val = format_data_csv(os.path.join(curr, "val_na_stratified.csv"), path)
    # decomp_all_files(
    #     val,
    #     local_path=os.path.join(path, "val"),
    #     n_workers=num_workers,
    #     format=".png",
    #     outside_prop=outside,
    #     inside_prop=inside,
    # )
