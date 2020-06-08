import matplotlib.pyplot as plt
import os



def plot_loss_in_dir_VAE(path, full_loss=True, generate_images=True, plot_total_L2=True, show_train_lines=False, save_path=None):
    os.chdir(path)
    FILE = f'ae_loss.dat'

    total_recon = []
    L2 = []
    actual_trajectories_L2 = []
    KL = []
    variance = []
    train_epochs = []

    data_dict = {}

    with open(FILE, "r") as f:
        for line in f.readlines():
            data = line.strip().split(",")
            total_recon.append(float(data[1]))
            L2.append(float(data[2]))
            KL.append(float(data[3]))
            variance.append(float(data[4]))
            actual_trajectories_L2.append(float(data[5]))
            if "IS_TRAIN" in data[-1]:
                # gen number, epochstrained / total
                train_epochs.append((int(data[0]), data[-2].strip()))

    if generate_images:
        f = plt.figure(figsize=(10, 5))

        spec = f.add_gridspec(2, 2)
        ax1 = f.add_subplot(spec[0, :])
        ax2 = f.add_subplot(spec[1, :])

        # L2 and variance on one plot
        if plot_total_L2:
            ax1.set_ylabel("L2")
            ax1.set_ylim([0, max(L2)])
            ln1 = ax1.plot(range(len(total_recon)), L2, c="red", label="L2 - Overall")
            ax1.annotate(f"{round(L2[-1],2)}", (len(total_recon) - 1, L2[-1]))

        ln2 = ax1.plot(range(len(actual_trajectories_L2)), actual_trajectories_L2, c="blue", label="L2 - Actual Trajectories")
        ax1.annotate(f"{round(actual_trajectories_L2[-1], 2)}", (len(actual_trajectories_L2) - 1, actual_trajectories_L2[-1]))

        if full_loss:
            var_ax = ax1.twinx()
            var_ax.set_ylabel("Variance")
            var_ax.set_ylim([0, max(variance)])
            ln3 = var_ax.plot(range(len(total_recon)), variance, c="green", label="Variance")
            var_ax.annotate(f"{round(variance[-1], 2)}", (len(variance) - 1, variance[-1]))

        # train marker
        if (show_train_lines):
            for (train_gen, train_ep) in train_epochs:
                ax1.axvline(train_gen, ls="--", lw=0.1, c="grey")
                ax2.axvline(train_gen, ls="--", lw=0.1, c="grey")

        # add in legends
        if plot_total_L2:
            lns = ln1+ln2+ln3 if full_loss else ln1+ln2
        else:
            lns = ln2 + ln3 if full_loss else ln2
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc='best')

        # aggregate loss and KL on one plot
        KL_ax = ax2.twinx()
        KL_ax.set_ylim([0, max(KL)])
        ax2.set_ylabel("Total Loss")
        KL_ax.set_ylabel("KL")
        ax2.set_ylim([min(total_recon), max(total_recon)])
        ln4 = ax2.plot(range(len(total_recon)), total_recon, c="red", label="Total Recon Loss")
        ln5 = KL_ax.plot(range(len(total_recon)), KL, c="blue", label="KL")

        # add in legends
        lns = ln4+ln5
        labs = [l.get_label() for l in lns]
        ax2.legend(lns, labs, loc='best')
        ax1.set_title(f"VAE Loss")
        ax2.set_xlabel("Generation")
        plt.savefig(f"vae_loss.png")
        plt.close()
        # plt.show()

    data_dict["TL"] = total_recon
    data_dict["L2"] = L2
    data_dict["AL"] = actual_trajectories_L2
    data_dict["KL"] = KL
    data_dict["VAR"] = variance
    data_dict["TR_EPOCHS"] = train_epochs
    return data_dict



if __name__ == "__main__":
    plot_loss_in_dir_VAE(
        "/home/andwang1/airl/balltrajectorysd/results_exp1/test/results_balltrajectorysd_vae/gen6001_random0.1_fulllosstrue/2020-06-06_07_16_34_6414")
