import matplotlib.pyplot as plt
import numpy as np
from exp_config import *
import os
import seaborn as sns


def plot_entropy_grid_in_dir(path, generate_images=True, save_path=None):
    os.chdir(path)
    files = os.listdir()

    # Find generation numbers
    generations = []
    for fname in files:
        if fname.startswith("entropy") and ".dat" in fname:
            generations.append(fname.rstrip(r".dat")[len("entropy"):])

    generations = sorted(int(gen) for gen in generations)

    # for plot across generations
    entropy_values = []
    entropy_values_exclzero = []
    percentage_moved = []

    for GEN_NUMBER in generations:
        if save_path:
            os.chdir(path)

        FILE_NAME = f'entropy{GEN_NUMBER}.dat'

        with open(FILE_NAME, "r") as f:
            lines = f.readlines()
            aggregate_stats = lines[2].strip().split(",")
            entropy = float(aggregate_stats[0])
            entropy_exclzero = float(aggregate_stats[1])
            moved = int(aggregate_stats[-1].split("/")[0])
            total = int(aggregate_stats[-1].split("/")[1])

            entropy_grid = lines[3].strip().split(",")
            freq_grid = lines[4].strip().split(",")

        entropy_values.append(entropy)
        entropy_values_exclzero.append(entropy_exclzero)
        percentage_moved.append(int(100 * moved / total))

        entropy_grid_values = []
        if generate_images:
            rows = []
            column = []
            rows_freq = []
            column_freq = []
            counter_x = 0
            for e, freq in zip(entropy_grid, freq_grid):
                if float(e) > -1:
                    entropy_grid_values.append(float(e))
                column.append(float(e))
                column_freq.append(int(freq))
                counter_x += 1
                if counter_x >= ENTROPY_DISCRETISATION:
                    counter_x = 0
                    rows.append(column)
                    column = []
                    rows_freq.append(column_freq)
                    column_freq = []

            fig = plt.figure(figsize=(15, 15))
            plt.ylim([ENTROPY_DISCRETISATION, 0])
            plt.xlim([0, ENTROPY_DISCRETISATION])

            # plot grid
            sns.heatmap(rows, vmin=-.5, vmax=np.log2(ENTROPY_DISCRETISATION ** 2) / 2, annot=rows_freq, linewidths=0.2,
                        cmap="coolwarm", fmt=".0f")

            plt.xticks(range(ENTROPY_DISCRETISATION), np.arange(0, ROOM_W, ROOM_W / ENTROPY_DISCRETISATION))
            plt.yticks(range(ENTROPY_DISCRETISATION), np.arange(0, ROOM_H, ROOM_H / ENTROPY_DISCRETISATION))
            plt.title(f"Entropy of Trajectories - Gen {GEN_NUMBER}")
            plt.xlabel("X")
            plt.ylabel("Y")

            ax1 = fig.add_subplot()
            textbox = f"Mean Entropy: {np.round(np.mean(entropy_values), 2)} / Excl. Zero: {np.round(np.mean(entropy_values_exclzero), 2)}"
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            # place a text box in upper left in axes coords
            ax1.text(0.787, 1.03, textbox, transform=ax1.transAxes, fontsize=12,
                     verticalalignment='top', bbox=props)

            if save_path:
                os.chdir(save_path)

            plt.savefig(f"entropy_{GEN_NUMBER}.png")
            plt.close()

    if generate_images:
        f = plt.figure(figsize=(10, 5))
        spec = f.add_gridspec(3, 1)
        ax1 = f.add_subplot(spec[:2, 0])
        ln1 = ax1.plot(generations, entropy_values, label="Mean Entropy", color="red", linestyle="--")
        ln2 = ax1.plot(generations, entropy_values_exclzero, label="Mean Entropy excl. 0s", color="red")
        ax1.set_ylabel("Mean Entropy")
        ax1.set_title("Entropy")

        lns = ln1 + ln2
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc='best')

        ax3 = f.add_subplot(spec[2, 0])
        ax3.set_title("% Solutions Moving The Ball")
        ax3.set_ylim([0, 100])
        ax3.set_yticks([0, 25, 50, 75, 100])
        ax3.yaxis.grid(True)
        ax3.plot(generations, percentage_moved, label="% solutions with impact")
        ax3.set_ylabel("%")
        ax3.set_xlabel("Generations")

        # make space between subplots
        plt.subplots_adjust(hspace=0.6)

        plt.savefig("entropy.png")
        plt.close()

    data_dict = {"gen": generations, "EV": entropy_values,
                 "EVE": entropy_values_exclzero, "PCT": percentage_moved}
    return data_dict


if __name__ == "__main__":
    plot_entropy_grid_in_dir(
        "/home/andwang1/airl/balltrajectorysd/results_box2d_exp1/first_run/results_balltrajectorysd_aurora/gen6001_random0_fulllossfalse_beta1_extension0_l2true/2020-06-22_18_14_25_195497")
