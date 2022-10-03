from lightml.data.decomp import DecompFromDataFrame

import pathlib 
import pandas as pd 
import os
import concurrent.futures 
import argparse
import tqdm 
import boto3
import datetime


curr = pathlib.Path(__file__).parent.resolve()
LOCAL_DOWNLOADED_VID_PATH = os.path.join(curr, "raw_data")

def convert_to_ms(time):
    # Time is formatted like: '1:03' or '01:03'
    time = time.replace(".", ":").split(":")
    m = int(time[0])
    s = int(time[1])
    ms = 1000 * (m*60 + s)
    print(ms)
    return 1000 * (m*60 + s)

def format_data_csv(path):
    df = pd.read_csv(path)

    # Name start_time / end_time as required by lightml decomp
    df = df.rename(columns={
        "entry-time": "start_time",
        "exit-time": "end_time",
        "s3_loc": "origin_uri",
    })

    # Format the s3 uris like in the standard db
    df["origin_uri"] = df["origin_uri"].apply(lambda x: os.path.join("https://s3.amazonaws.com/", x))

    # Name the "value" (label) as in_procedure, since we are just doing a binary problem
    df["value"] = ["in_procedure"]*len(df)

    # Define our local path to download uris to 
    df["local_path"] = df["origin_uri"].apply(lambda x: os.path.join(LOCAL_DOWNLOADED_VID_PATH, x.split('/')[-1]))

    # Remove videos that start/end inside colon
    df = df.dropna(subset=["start_time", "end_time"])

    # Convert start time from MM:SS to milliseconds for lightml decomp
    df["start_time"] = df["start_time"].apply(lambda x: convert_to_ms(x))
    df["end_time"] = df["end_time"].apply(lambda x: convert_to_ms(x))

    return df


if __name__ == "__main__":
    os.makedirs(LOCAL_DOWNLOADED_VID_PATH, exist_ok=True)

    # Format the csv files for use with lightml
    train = format_data_csv(os.path.join(curr, 'training_data.csv'))
    holdout = format_data_csv(os.path.join(curr, 'holdout_data.csv'))
    print(train.apply(lambda x: x["start_time"] < x["end_time"], axis=1))
    # perform decomp for each video
    handler = DecompFromDataFrame(
        files=train,
        frames_per_clip=100,
        frames_per_nonclip=100,
        local_path=os.path.join(curr, 'decompV1'),
        img_format=".jpg",
        val_prop=0.3,
        test_prop=0,
    )

    handler.decomp()
