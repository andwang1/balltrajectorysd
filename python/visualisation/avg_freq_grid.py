import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from exp_config import *
import os

# make font bigger
plt.rc('font', size=14)

def plot_avg_freq_grid(path, generate_images=True, save_path=None):
    os.chdir(path)
    experiments = [dir for dir in os.listdir() if os.path.isdir(dir)]
    div_grids = np.zeros((DISCRETISATION, DISCRETISATION))
    freq_grids = np.zeros((DISCRETISATION, DISCRETISATION))
    num_solutions = 0
    for exp in experiments:
        print(exp)
        os.chdir(os.path.join(path, exp))
        files = os.listdir()

        # Find generation numbers
        div_generations = []
        for fname in files:
            if fname.startswith("diversity") and ".dat" in fname:
                div_generations.append(fname.rstrip(r".dat")[len("diversity"):])

        div_generations = sorted(int(gen) for gen in div_generations)

        GEN_NUMBER = div_generations[-1]

        FILE_NAME = f'diversity{GEN_NUMBER}.dat'

        with open(FILE_NAME, "r") as f:
            lines = f.readlines()
            max_diversity = int(lines[0].strip().split(":")[-1])
            achieved_diversity = round(float(lines[1].strip()), 5)
            # bitmap prints in reverse order
            diversity_grid = lines[2].strip().split(",")[::-1]

        FILE_NAME = f'similarities{GEN_NUMBER}.dat'

        with open(FILE_NAME, "r") as f:
            lines = f.readlines()
            freq_grid = lines[4].strip().split(",")

        diversity_grid = np.array([float(i) for i in diversity_grid]).reshape((DISCRETISATION, DISCRETISATION))
        div_grids += diversity_grid
        freq_grid = np.array([int(i) for i in freq_grid]).reshape((DISCRETISATION, DISCRETISATION))
        freq_grids += freq_grid
        num_solutions += np.sum(freq_grid)


    # Average
    div_grids /= len(experiments)
    freq_grids /= len(experiments)
    num_solutions /= len(experiments)

    # plot colours
    fig = plt.figure(figsize=(15, 15))
    plt.ylim([DISCRETISATION, 0])
    plt.xlim([0, DISCRETISATION])

    # vmin/vmax sets limits
    # color = plt.pcolormesh(div_grids, vmin=0, vmax=1)

    # plot grid
    plt.grid(which="both")
    sns.heatmap(freq_grids, vmin=0, vmax=170, annot=freq_grids, linewidths=0.2, fmt=".0f")

    plt.xticks(range(DISCRETISATION), np.arange(0, ROOM_W, ROOM_W / DISCRETISATION))
    plt.yticks(range(DISCRETISATION), np.arange(0, ROOM_H, ROOM_H / DISCRETISATION))

    # fig.colorbar(color)
    plt.title(f"Average number of trajectories ending in bin - Gen {GEN_NUMBER}")
    plt.xlabel("X")
    plt.ylabel("Y")

    if save_path:
        os.chdir(save_path)
    os.chdir(path)
    plt.savefig(f"freq_avg_{GEN_NUMBER}.png")
    plt.savefig(f"freq_avg_{GEN_NUMBER}.pdf")
    plt.close()

if __name__ == "__main__":
    plot_avg_freq_grid(
        # "/media/andwang1/SAMSUNG/MSC_INDIV/results_box2d_bsd_exp1/sne_nosampletrain_beta0/results_balltrajectorysd_vae/gen6001_random1_fulllosstrue_beta0_extension0_lossfunc2_samplefalse_tsne1")
"/media/andwang1/SAMSUNG/MSC_INDIV/results_box2d_bsd_exp1/l2nosampletrain/results_balltrajectorysd_vae/gen6001_random1_fulllosstrue_beta1_extension0_lossfunc2_samplefalse")
