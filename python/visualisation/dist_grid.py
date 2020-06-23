import matplotlib.pyplot as plt
import numpy as np
from exp_config import *
import os


def plot_dist_grid_in_dir(path, generate_images=True, save_path=None):
    os.chdir(path)
    files = os.listdir()

    # Find generation numbers
    dist_generations = []
    for fname in files:
        if fname.startswith("distances") and ".dat" in fname:
            dist_generations.append(fname.rstrip(r".dat")[len("distances"):])

    dist_generations = sorted(int(gen) for gen in dist_generations)

    # for distance plot across generations
    dist_values = []
    dist_values_exclzero = []
    var_values = []
    var_values_exclzero = []
    percentage_moved = []

    for GEN_NUMBER in dist_generations:
        if save_path:
            os.chdir(path)

        FILE_NAME = f'distances{GEN_NUMBER}.dat'

        with open(FILE_NAME, "r") as f:
            lines = f.readlines()
            mean_distance = float(lines[2].strip().split(",")[0])
            var_distance = float(lines[2].strip().split(",")[1])
            mean_distance_exclzero = float(lines[2].strip().split(",")[2])
            var_distance_exclzero = float(lines[2].strip().split(",")[3])
            moved = int(lines[2].strip().split(",")[-1].split("/")[0])
            total = int(lines[2].strip().split(",")[-1].split("/")[1])
            # bitmap prints in reverse order
            distance_grid = lines[3].strip().split(",")
            min_max_grid = lines[4].strip().split(",")

        dist_values.append(mean_distance)
        dist_values_exclzero.append(mean_distance_exclzero)
        var_values.append(var_distance)
        var_values_exclzero.append(var_distance_exclzero)
        percentage_moved.append(int(100 * moved / total))

        if generate_images:
            rows = []
            column = []
            counter_x = 0
            for i in distance_grid:
                column.append(float(i))
                counter_x += 1
                if counter_x >= DISCRETISATION:
                    counter_x = 0
                    rows.append(column)
                    column = []

            # variance grid
            # plot colours
            fig = plt.figure(figsize=(15, 15))
            plt.ylim([DISCRETISATION, 0])
            plt.xlim([0, DISCRETISATION])

            # vmin/vmax sets limits
            color = plt.pcolormesh(rows, vmin=-10, vmax=50)

            # plot grid
            plt.grid(which="both")
            plt.xticks(range(DISCRETISATION), np.arange(0, ROOM_W, ROOM_W / DISCRETISATION))
            plt.yticks(range(DISCRETISATION), np.arange(0, ROOM_H, ROOM_H / DISCRETISATION))
            fig.colorbar(color)
            plt.title(f"Variances of Distances - Gen {GEN_NUMBER} - Total Var. {round(var_distance_exclzero, 2)}")
            plt.xlabel("X")
            plt.ylabel("Y")

            ax1 = fig.add_subplot()
            textbox = f"Mean: {round(mean_distance_exclzero, 2)} / Var: {round(var_distance_exclzero, 2)}"
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            # place a text box in upper left in axes coords
            ax1.text(0.787, 1.03, textbox, transform=ax1.transAxes, fontsize=12,
                     verticalalignment='top', bbox=props)

            if save_path:
                os.chdir(save_path)

            plt.savefig(f"distance_var_{GEN_NUMBER}.png")
            plt.close()

            # minmax grid
            rows = []
            column = []
            counter_x = 0
            for i in min_max_grid:
                column.append(float(i))
                counter_x += 1
                if counter_x >= DISCRETISATION:
                    counter_x = 0
                    rows.append(column)
                    column = []

            # plot colours
            fig = plt.figure(figsize=(15, 15))
            plt.ylim([DISCRETISATION, 0])
            plt.xlim([0, DISCRETISATION])

            # vmin/vmax sets limits
            color = plt.pcolormesh(rows, vmin=-ROOM_W, vmax=ROOM_W * 4)

            # plot grid
            plt.grid(which="both")
            plt.xticks(range(DISCRETISATION), np.arange(0, ROOM_W, ROOM_W / DISCRETISATION))
            plt.yticks(range(DISCRETISATION), np.arange(0, ROOM_H, ROOM_H / DISCRETISATION))
            fig.colorbar(color)
            plt.title(f"Max - Min of Distances - Gen {GEN_NUMBER}")
            plt.xlabel("X")
            plt.ylabel("Y")

            ax1 = fig.add_subplot()
            if save_path:
                os.chdir(save_path)

            plt.savefig(f"distance_minmax_{GEN_NUMBER}.png")
            plt.close()

    if generate_images:
        f = plt.figure(figsize=(10, 5))
        spec = f.add_gridspec(3, 1)
        ax1 = f.add_subplot(spec[:2, 0])
        ln1 = ax1.plot(dist_generations, dist_values, label="Mean Distance", color="red", linestyle="--")
        ln2 = ax1.plot(dist_generations, dist_values_exclzero, label="Mean Distance excl. 0s", color="red")
        ax1.set_ylabel("Mean Distance")

        ax2 = ax1.twinx()
        ln3 = ax2.plot(dist_generations, var_values, label="Variance", color="blue", linestyle="--")
        ln4 = ax2.plot(dist_generations, var_values_exclzero, label="Variance excl. 0s", color="blue")
        ax2.set_ylabel("Variance")
        ax2.set_title("Distance Stats over Generations")

        lns = ln1 + ln2 + ln3 + ln4
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc='best')

        ax3 = f.add_subplot(spec[2, 0])
        ax3.set_title("% Solutions Moving The Ball")
        ax3.set_ylim([0, 100])
        ax3.set_yticks([0, 25, 50, 75, 100])
        ax3.yaxis.grid(True)
        ax3.plot(dist_generations, percentage_moved, label="% solutions with impact")
        ax3.set_ylabel("%")
        ax3.set_xlabel("Generations")

        # make space between subplots
        plt.subplots_adjust(hspace=0.6)

        plt.savefig("distance.png")
        plt.close()

    data_dict = {"gen": dist_generations, "MD": mean_distance, "MDE": mean_distance_exclzero, "VD": var_values,
                 "VDE": var_values_exclzero, "PCT": percentage_moved}
    return data_dict


if __name__ == "__main__":
    plot_dist_grid_in_dir(
        "/home/andwang1/airl/balltrajectorysd/results_box2d_exp1/box2dtest/vistest/2020-06-19_19_12_49_126106")
