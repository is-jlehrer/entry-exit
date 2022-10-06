# Gross but just temporary
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pathlib

from lightml.data.decomp import DecompFromDataFrame
from utils import format_data_csv, DECOMP_PATH

curr = pathlib.Path(__file__).parent.resolve()

if __name__ == "__main__":
    # perform decomp for each video
    handler = DecompFromDataFrame(
        files=format_data_csv(os.path.join(curr, 'combined_data.csv')),
        frames_per_clip=500,
        frames_per_nonclip=500,
        local_path=os.path.join(DECOMP_PATH),
        img_format=".jpg",
        val_prop=0.2,
        test_prop=0.05,
        max_workers=8,
    )

    handler.decomp()
