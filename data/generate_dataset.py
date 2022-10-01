from lightml.data.decomp import DecompFromDataFrame
from lightml.common.defaults import download_from_uri

import pathlib 
import pandas as pd 
import os
import concurrent.futures 
import argparse
import tqdm 
import boto3

curr = pathlib.Path(__file__).parent.resolve()
LOCAL_DOWNLOADED_VID_PATH = os.path.join(curr, "raw_data")

def download_from_uri(uri, local_path):
    # New client since these objects arent picklable and therefore 
    # cannot be used with concurrent.futures

    client = boto3.client("s3")
    bucket = uri.split('/')[0]
    url = '/'.join(uri.split('/')[1:])

    if not os.path.isfile(local_path):
        client.download_file(bucket, url, local_path)

def download_dataset(df):
    # Download the videos in total before decomp (we usually delete them as frames are extracted) 
    Because we need to get the fps to calculate sampling rates for decomp
    with tqdm.tqdm(desc="Downloading PHI videos for entry/exit decomp", total=len(df)) as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            for uri, local_path in zip(df["origin_uri"].values, df["local_path"].values):
                # More inefficient than apply but still < 200 vids generally. Slowdown is marginal
                futures.append(
                    executor.submit(
                        download_from_uri,
                        uri=uri,
                        local_path=local_path,
                    )
                )
            for future in concurrent.futures.as_completed(futures):
                pbar.update(1)
                if future.exception():
                    print(f"Exception at {future}, {future.exception()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download S3 URIs and perform frame-level decomp using LightML decomp")
    parser.add_argument('--feature', action='store_true')
    parser.add_argument('--no-feature', dest='feature', action='store_false')
    
    # Make the folder to download videos to, if it doesn't exist already
    os.makedirs(LOCAL_DOWNLOADED_VID_PATH, exist_ok=True)
    # Read in the training csv 
    df = pd.read_csv(os.path.join(curr, 'training_data.csv'))

    # Name start_time / end_time as required by lightml decomp
    df = df.rename(columns={
        "entry-time": "start_time",
        "exit-time": "end_time",
        "s3_loc": "origin_uri",
    })

    # Name the "value" (label) as in_procedure, since we are just doing a binary problem
    df["value"] = ["in_procedure"]*len(df)

    # Define our local path to download uris to 
    df["local_path"] = df["origin_uri"].apply(lambda x: os.path.join(LOCAL_DOWNLOADED_VID_PATH, x.split('/')[-1]))


    decomp = DecompFromDataFrame(
        files=df,
        frames_per_clip=100,
        frames_per_nonclip=100,
        local_path=os.path.join(curr, 'decompV1'),
        img_format=".jpg",
        val_prop=0.3,
        test_prop=0,
    )

