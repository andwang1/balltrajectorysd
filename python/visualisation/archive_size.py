import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

path = "/media/andwang1/SAMSUNG/MSC_INDIV/results_box2d_bsd_benchmarks/exclalgo/results_balltrajectorysd_benchmarks_exclude_algo"


os.chdir(path)
dirs = [dir for dir in os.listdir() if os.path.isdir(dir)]
archive_sizes = []
stochasticities = []
for dir in dirs:
    os.chdir(path)
    component = dir.split("_")
    for part in component:
        if "random" in part:
            stochasticity = part[len("random"):]
            break

    os.chdir(dir)
    PIDs = [dir for dir in os.listdir() if os.path.isdir(dir)]
    for p in PIDs:
        with open(f"{p}/archive_6000.dat", "r") as f:
            stochasticities.append(stochasticity)
            archive_sizes.append(len(f.readlines()))

# make legend bigger
plt.rc('legend', fontsize=35)
# make lines thicker
plt.rc('lines', linewidth=4, linestyle='-.')
# make font bigger
plt.rc('font', size=30)
sns.set_style("dark")

f = plt.figure(figsize=(20, 20))
spec = f.add_gridspec(1, 2)
ax1 = f.add_subplot(spec[0, :])

sns.lineplot(stochasticities, archive_sizes, estimator=np.median, ci=None, ax=ax1)
data_stats = pd.DataFrame({"stoch":stochasticities, "size":archive_sizes})[["stoch", "size"]].groupby("stoch").describe()
quart25 = data_stats[('size', '25%')]
quart75 = data_stats[('size', '75%')]
ax1.fill_between([0, 1, 2, 3, 4, 5], quart25, quart75, alpha=0.3)
ax1.set_title("Archive Size")
ax1.set_ylabel("Size")
ax1.set_xlabel("Stochasticity")
os.chdir(path)
plt.savefig("excl_algo_archive_size.pdf")
plt.show()