import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle as pk
import numpy as np
import pandas as pd

sns.set_style("dark")

PATH = "/media/andwang1/SAMSUNG/MSC_INDIV/results_box2d_bsd_exp1/l2nosampletrain_extend50/results_balltrajectorysd_vae"
FILE_NAME = "ae_loss.dat"

experiments = [exp_name for exp_name in os.listdir(PATH) if os.path.isdir(os.path.join(PATH, exp_name))]

percentage = []
stochasticities = []

for exp in experiments:
    os.chdir(PATH)
    os.chdir(exp)
    stoch = exp.split("_")[1][len("random"):]
    PIDs = [pid for pid in os.listdir() if os.path.isdir(pid)]
    for pid in PIDs:
        os.chdir(os.path.join(PATH, exp, pid))
        print(os.getcwd())
        with open(FILE_NAME, "r") as f:
            data = f.readlines()[10::10]
        for line in data:
            percentage.append(float(line.split(",")[-3]))
            stochasticities.append(stoch)

stochasticities = np.array(stochasticities).flatten()
percentage = np.round(np.array(percentage).flatten() * 100, 0)
os.chdir(PATH)
f = plt.figure(figsize=(10, 5))
spec = f.add_gridspec(1, 1)
ax1 = f.add_subplot(spec[0, 0])
ln1 = sns.lineplot(stochasticities, percentage, estimator=np.median, ci=None, ax=ax1, color="blue")
data_stats = pd.DataFrame({"x": stochasticities, "y": percentage}).groupby("x").describe()
quart25 = data_stats[("y", '25%')]
quart75 = data_stats[("y", '75%')]
ax1.fill_between([0, 1, 2, 3, 4, 5], quart25, quart75, alpha=0.3, color="blue")
plt.title("% of Solutions affected by Noise in Extended Dataset")
ax1.set_ylabel("%")
ax1.set_xlabel("Stochasticity")
plt.savefig("extend_pct_random.pdf")
# plt.show()