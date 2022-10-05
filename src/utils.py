import os
import pandas as pd
import pathlib 

curr = pathlib.Path(__file__).parent.resolve()
DECOMP_PATH = os.path.join(curr, 'decomp_400_frames_balanced')

def convert_to_ms(time):
    # Time is formatted like: '1:03' or '01:03'
    time = time.replace(".", ":").split(":")
    m = int(time[0])
    s = int(time[1])
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
    df["value"] = ["procedure"]*len(df)

    # Define our local path to download uris to 
    df["local_path"] = df["origin_uri"].apply(lambda x: os.path.join(DECOMP_PATH, x.split('/')[-1]))

    # Remove videos that start/end inside colon
    df = df.dropna(subset=["start_time", "end_time"])
    
    # Convert start time from MM:SS to milliseconds for lightml decomp
    df["start_time"] = df["start_time"].apply(lambda x: convert_to_ms(x))
    df["end_time"] = df["end_time"].apply(lambda x: convert_to_ms(x))

    return df


