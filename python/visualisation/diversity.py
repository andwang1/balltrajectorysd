import matplotlib.pyplot as plt
import numpy as np
from exp_config import *
import os


def plot_diversity_in_dir(path, generate_images=True, save_path=None):
    os.chdir(path)
    files = os.listdir()

    # Find generation numbers
    div_generations = []
    for fname in files:
        if fname.startswith("diversity") and ".dat" in fname:
            div_generations.append(fname.rstrip(r".dat")[len("diversity"):])

    div_generations = sorted(int(gen) for gen in div_generations)

    # for diversity plot across generations
    div_scores = []

    for GEN_NUMBER in div_generations:
        if save_path:
            os.chdir(path)

        FILE_NAME = f'diversity{GEN_NUMBER}.dat'

        with open(FILE_NAME, "r") as f:
            lines = f.readlines()
            max_diversity = int(lines[0].strip().split(":")[-1])
            achieved_diversity = round(float(lines[1].strip()), 5)
            # bitmap prints in reverse order
            diversity_grid = lines[2].strip().split(",")[::-1]

        div_scores.append(achieved_diversity)

        rows = []
        column = []
        counter_x = 0
        for i in diversity_grid:
            column.append(float(i))
            counter_x += 1
            if counter_x >= DISCRETISATION:
                counter_x = 0
                rows.append(column)
                column = []

        if generate_images:
            # plot colours
            fig = plt.figure(figsize=(15, 15))
            plt.ylim([DISCRETISATION, 0])
            plt.xlim([0, DISCRETISATION])

            # vmin/vmax sets limits
            color = plt.pcolormesh(rows, vmin=0, vmax=1)

            # plot grid
            plt.grid(which="both")
            plt.xticks(range(DISCRETISATION), np.arange(0, ROOM_W, ROOM_W / DISCRETISATION))
            plt.yticks(range(DISCRETISATION), np.arange(0, ROOM_H, ROOM_H / DISCRETISATION))
            fig.colorbar(color)
            plt.title(f"Diversity - BinsTransversed / TotalBins - Gen {GEN_NUMBER}")
            plt.xlabel("X")
            plt.ylabel("Y")

            ax1 = fig.add_subplot()
            textbox = f"Score: {achieved_diversity} of {max_diversity}"
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            # place a text box in upper left in axes coords
            ax1.text(0.787, 1.03, textbox, transform=ax1.transAxes, fontsize=12,
                     verticalalignment='top', bbox=props)

            if save_path:
                os.chdir(save_path)

            plt.savefig(f"diversity_{GEN_NUMBER}.png")
            plt.close()

    if generate_images:
        plt.plot(div_generations, div_scores, label="Diversity")
        plt.xlabel("Generations")
        plt.ylabel("Diversity")
        plt.title("Diversity over Generations")
        plt.hlines(max_diversity, 0, div_generations[-1], linestyles="--", label="Max Diversity")
        plt.legend(loc=4)
        plt.savefig("diversity.png")
        plt.close()

    return {gen: score for gen, score in zip(div_generations, div_scores)}, max_diversity


if __name__ == "__main__":
    plot_diversity_in_dir(
        "/home/andwang1/airl/balltrajectorysd/results_exp1/repeated_run1/results_balltrajectorysd_ae/--number-gen=6001_--pct-random=0.2_--full-loss=false/2020-06-05_02_56_35_224997")
