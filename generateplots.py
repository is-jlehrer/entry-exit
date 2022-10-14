import matplotlib.pyplot as plt
import pandas as pd
import os

plotfolder = "resnet18_500kdecomp_plots_smoothed"
os.makedirs(plotfolder, exist_ok=True)
df = pd.read_csv("model_results_sample_rate_5_resnet18_500k_decomp.csv", index_col="Unnamed: 0")
df = df.T  # Now we can do moving avg of columns

for i in df.columns:
    r = df.loc[:, i].rolling(window=15).mean()
    plt.scatter(list(range(len(r))), r, s=1)
    plt.savefig(f"{plotfolder}/row_{i}.png")
    plt.clf()
