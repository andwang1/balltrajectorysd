import matplotlib.pyplot as plt
import os

SHOW_TRAIN_LINES = False

variant = "vae"
random = "0.2"
GEN_NUMBER = 6000

vae_loss = "fulllosstrue"

BASE_PATH = '/home/andwang1/airl/balltrajectorysd/results_exp1/second_run/'
EXP_PATH = f'results_balltrajectorysd_{variant}/gen6001_random{random}_{vae_loss}/'
os.chdir(BASE_PATH+EXP_PATH)
PID = os.listdir()[0] + "/"
os.chdir(BASE_PATH)
FILE_NAME = f'ae_loss.dat'

FILE = BASE_PATH + EXP_PATH + PID + FILE_NAME

# ofs << ea.gen() << ", " << recon << ", " << L2 << ", " << KL << ", " << var;

total_recon = []
L2 = []
KL = []
variance = []
train_epochs = []

with open(FILE, "r") as f:
    for line in f.readlines():
        data = line.strip().split(",")
        total_recon.append(float(data[1]))
        L2.append(float(data[2]))
        KL.append(float(data[3]))
        variance.append(float(data[4]))
        if "IS_TRAIN" in data[-1]:
            # gen number, epochstrained / total
            train_epochs.append((int(data[0]), data[-2].strip()))


f = plt.figure(figsize=(10, 5))

spec = f.add_gridspec(2, 2)
# both kwargs together make the box squared
ax1 = f.add_subplot(spec[0, :])
ax2 = f.add_subplot(spec[1, :])

# L2 and variance on one plot
ax1.set_ylabel("L2")
ax1.set_ylim([0, max(L2)])
ln1 = ax1.plot(range(len(total_recon)), L2, c="red", label="L2")
ax1.annotate(f"{round(L2[-1],2)}", (len(total_recon) - 1, L2[-1]))

if vae_loss == "fulllosstrue":
    var_ax = ax1.twinx()
    var_ax.set_ylabel("Variance")
    var_ax.set_ylim([0, max(variance)])
    ln2 = var_ax.plot(range(len(total_recon)), variance, c="blue", label="Variance")
    var_ax.annotate(f"{round(variance[-1], 2)}", (len(variance) - 1, variance[-1]))

# train marker
if (SHOW_TRAIN_LINES):
    for (train_gen, train_ep) in train_epochs:
        ax1.axvline(train_gen, ls="--", lw=0.1, c="grey")
        ax2.axvline(train_gen, ls="--", lw=0.1, c="grey")

# add in legends

lns = ln1+ln2 if vae_loss == "fulllosstrue" else ln1
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc='best')

# aggregate loss and KL on one plot
KL_ax = ax2.twinx()
ax2.set_ylabel("Total Loss")
KL_ax.set_ylabel("KL")
ax2.set_ylim([min(total_recon), max(total_recon)])
KL_ax.set_ylim([0, max(KL)])
ln3 = ax2.plot(range(len(total_recon)), total_recon, c="red", label="Total Recon Loss")
ln4 = KL_ax.plot(range(len(total_recon)), KL, c="blue", label="KL")

# add in legends
lns = ln3+ln4
labs = [l.get_label() for l in lns]
ax2.legend(lns, labs, loc='best')
ax1.set_title(f"VAE Loss")

plt.savefig(f"vae_loss_{random}_{vae_loss}.pdf")
# plt.show()