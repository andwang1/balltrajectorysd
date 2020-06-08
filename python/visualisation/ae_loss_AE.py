import matplotlib.pyplot as plt
import os


def plot_loss_in_dir_AE(path, generate_images=True, is_aurora=False, show_train_lines=False, save_path=None):
    os.chdir(path)
    FILE = f'ae_loss.dat'

    total_recon = []
    actual_trajectories_L2 = []
    train_epochs = []

    data_dict = {}

    with open(FILE, "r") as f:
        for line in f.readlines():
            data = line.strip().split(",")
            total_recon.append(float(data[1]))
            if not is_aurora:
                actual_trajectories_L2.append(float(data[2]))
            if "IS_TRAIN" in data[-1]:
                # gen number, epochstrained / total
                train_epochs.append((int(data[0]), data[-2].strip()))

    if generate_images:
        f = plt.figure(figsize=(10, 5))

        spec = f.add_gridspec(1, 1)
        # both kwargs together make the box squared
        ax1 = f.add_subplot(spec[0, 0])

        ax1.set_ylabel("L2")
        ax1.set_ylim([0, max(total_recon)])
        ln1 = ax1.plot(range(len(total_recon)), total_recon, c="red", label="L2 - Overall")
        ax1.annotate(f"{round(total_recon[-1], 2)}", (len(total_recon) - 1, total_recon[-1]))

        if not is_aurora:
            ln2 = ax1.plot(range(len(actual_trajectories_L2)), actual_trajectories_L2, c="blue", label="L2 - Actual Trajectories")
            ax1.annotate(f"{round(actual_trajectories_L2[-1], 2)}", (len(actual_trajectories_L2) - 1, actual_trajectories_L2[-1]))

        # train marker
        if (show_train_lines):
            for (train_gen, train_ep) in train_epochs:
                ax1.axvline(train_gen, ls="--", lw=0.1, c="grey")

        # add in legends
        lns = ln1 + ln2 if not is_aurora else ln1
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc='best')

        ax1.set_title(f"AE Loss")
        ax1.set_xlabel("Generation")
        plt.savefig(f"ae_loss.png")
        plt.close()

    data_dict["L2"] = total_recon
    data_dict["TR_EPOCHS"] = train_epochs
    if not is_aurora:
        data_dict["AL"] = actual_trajectories_L2
    return data_dict

if __name__ == "__main__":
    plot_loss_in_dir_AE(
        "/home/andwang1/airl/balltrajectorysd/results_exp1/repeated_run1/results_balltrajectorysd_ae/gen6001_random0.1_fulllossfalse/2020-06-04_23_21_36_261296")
