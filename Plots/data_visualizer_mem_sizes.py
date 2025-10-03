# %%
import pickle as pkl
import glob
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

gloober = glob.glob("results/grid_search34/*") 

row1 = 3

fig = plt.figure(figsize=(2*row1,4))

axs = []

for i in range(row1):
     axs.append(plt.subplot2grid((2,2*row1), (0,2*i), colspan=2))

for i in range(row1-1):
     axs.append(plt.subplot2grid((2,2*row1), (1,2*i+1), colspan=2))

best_acc_yet = [0, 0]

list_to_rearange = [4,0,1,7,3]

rearange = [0,0,0,0,0]

for i in range(5):
    rearange[i] = gloober[list_to_rearange[i]]

gloober = rearange

# Define a custom formatter to round tick labels
def round_to_decimal(x, pos):
    return f"{round(x, 2)}"  # Round to 2 decimal places

# Apply the custom formatter
formatter = FuncFormatter(round_to_decimal)



for idx, glob in enumerate(gloober):

    with open(glob, "rb") as f:
        results = pkl.load(f)
        print(len(results["Model Parameters"][0][1][0]))
        axs[idx].plot(results["Accuracy Measurements"]["Training accuracy"], label="Train")
        axs[idx].plot(results["Accuracy Measurements"]["Testing accuracy"], label="Test")
        axs[idx].grid()
        axs[idx].set_title(f"Mem size : {len(results['Model Parameters'][0][1][0])}")
        axs[idx].legend()
        axs[idx].yaxis.set_major_formatter(formatter)
        print(np.max(results["Accuracy Measurements"]["Testing accuracy"]))
        if np.max(results["Accuracy Measurements"]["Testing accuracy"]) > best_acc_yet[0]:
                best_acc_yet[0] = np.max(results["Accuracy Measurements"]["Testing accuracy"])
                best_acc_yet[1] = glob

plt.tight_layout()
plt.savefig("figures/Different_mem_sizes.png")


