import matplotlib.pyplot as plt
import os

SHOW_TRAIN_LINES = False

variant = "ae"
random = "0.2"
GEN_NUMBER = 6000

BASE_PATH = '/home/andwang1/airl/balltrajectorysd/results_exp1/second_run/'
EXP_PATH = f'results_balltrajectorysd_{variant}/gen6001_random{random}/'
os.chdir(BASE_PATH+EXP_PATH)
PID = os.listdir()[0] + "/"
os.chdir(BASE_PATH)
FILE_NAME = f'ae_loss.dat'

FILE = BASE_PATH + EXP_PATH + PID + FILE_NAME

# ofs << ea.gen() << ", " << recon << ", " << L2 << ", " << KL << ", " << var;

total_recon = []
train_epochs = []

with open(FILE, "r") as f:
    for line in f.readlines():
        data = line.strip().split(",")
        total_recon.append(float(data[1]))
        if "IS_TRAIN" in data[-1]:
            # gen number, epochstrained / total
            train_epochs.append((int(data[0]), data[-2].strip()))


f = plt.figure(figsize=(10, 5))

spec = f.add_gridspec(1, 1)
# both kwargs together make the box squared
ax1 = f.add_subplot(spec[0, 0])

# L2 and variance on one plot
ax1.set_ylabel("L2")
ax1.set_ylim([0, max(total_recon)])
ln1 = ax1.plot(range(len(total_recon)), total_recon, c="red", label="L2")
ax1.annotate(f"{round(total_recon[-1],2)}", (len(total_recon) - 1, total_recon[-1]))

# train marker
if (SHOW_TRAIN_LINES):
    for (train_gen, train_ep) in train_epochs:
        ax1.axvline(train_gen, ls="--", lw=0.1, c="grey")

# add in legends

labs = [l.get_label() for l in ln1]
ax1.legend(ln1, labs, loc='best')

ax1.set_title(f"AE Loss")

plt.savefig(f"ae_loss_{random}.pdf")
plt.show()