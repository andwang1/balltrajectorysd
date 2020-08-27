import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle as pk
import numpy as np
import pandas as pd

colours = ["blue", "brown", "grey", "green", "purple", "red", "pink", "orange"]
sns.set_style("dark")

PATH = "/media/andwang1/SAMSUNG/MSC_INDIV/results_box2d_bsd_exp1/l2nosampletrain_extend50_dist/results_balltrajectorysd_vae/"
FILE_NAME = "ae_loss.dat"

experiments = [exp_name for exp_name in os.listdir(PATH) if os.path.isdir(os.path.join(PATH, exp_name))]

f = plt.figure(figsize=(10, 5))
spec = f.add_gridspec(1, 1)
ax1 = f.add_subplot(spec[0, 0])

colour_count = 0
for exp in experiments:
    os.chdir(PATH)
    os.chdir(exp)

    train_iter = []
    dist = []

    PIDs = [pid for pid in os.listdir() if os.path.isdir(pid)]
    for pid in PIDs:
        os.chdir(os.path.join(PATH, exp, pid))
        print(os.getcwd())
        with open(FILE_NAME, "r") as f:
            data = f.readlines()[10::10]
        for i, line in enumerate(data):
            dist.append(float(line.split(",")[-3]))
            train_iter.append(i + 1)

    sns.lineplot(train_iter, dist, estimator=np.median, ci=None, ax=ax1, label=exp, color=colours[colour_count])
    data_stats = pd.DataFrame({"x": train_iter, "y": dist}).groupby("x").describe()
    quart25 = data_stats[("y", '25%')]
    quart75 = data_stats[("y", '75%')]
    ax1.fill_between(list(range(600)), quart25, quart75, alpha=0.3, color="blue")
    colour_count += 1

plt.title("Mean Distance in Extension Dataset")
ax1.set_ylabel("Distance")
ax1.set_xlabel("Train Iteration")
plt.legend(title="Stochasticity")
os.chdir(PATH)
plt.savefig("extend_pct_random.png")
plt.savefig("extend_pct_random.pdf")
# plt.savefig("dist_extension.png")
plt.show()