import os
import sys
import re
import numpy as np
import pandas as pd
from mpl_toolkits import mplot3d
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plot


def main(directory):
    results = {
        "fs": [],
        "gts": [],
        "gain": []
    }
    for root, dirs, files in os.walk(directory):
        for f in files:
            if f.endswith(".csv"):
                file_path = os.path.join(root, f)
                regex_match = re.findall("[0-9]*\.[0-9]+", f)
                df = pd.read_csv(file_path)
                df["gain"] = df["gt_fedavg_acc"] - df["fedavg_acc"]
                results["fs"].append(
                    float(regex_match[0])
                )
                results["gts"].append(
                    float(regex_match[1])
                )
                results["gain"].append(
                    df["gain"].mean()
                )
    df_results = pd.DataFrame(results)

    # Plot 3D graph
    x = df_results["fs"].to_numpy()
    y = df_results["gts"].to_numpy()
    z = df_results["gain"].to_numpy()
    axes = plot.axes(projection='3d')
    axes.plot_trisurf(x,y,z, cmap='viridis')
    axes.scatter3D(x,y,z, color='red')
    axes.set_xlabel('foreign data split ratio')
    axes.set_ylabel('game theory fed avg test data split ratio')
    axes.set_zlabel('gain in accuracy')
    axes.set_title('Gain in accuracy for decentralized federated learning with game theory')
    plot.show()


if __name__ == "__main__":
    directory = str(sys.argv[1])
    main(directory)
