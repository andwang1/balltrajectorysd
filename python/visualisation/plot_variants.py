import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle as pk

path = "/media/andwang1/SAMSUNG/MSC_INDIV/results_box2d_bsd_exp1"
os.chdir(path)

plotting_groups = [
    ["l1", "smoothl1"]
]

colours = ["blue", "red", "green", "brown", "yellow", "purple", "orange"]

for group in plotting_groups:
    f = plt.figure(figsize=(5, 5))
    spec = f.add_gridspec(1, 2)
    ax1 = f.add_subplot(spec[0, :])
    colour_count = 0
    for member in group:
        with open(f"{member}/loss_data.pk", "rb") as f:
            loss_data = pk.load(f)

        for variant, data in loss_data.items():
            if "aurora" in variant:
                continue
            sns.lineplot(data["stoch"], data["AL"], estimator="mean", ci="sd", label=f"{member}-{variant}", ax=ax1, color=colours[colour_count])
            colour_count += 1
    ax1.set_title("Losses - Actual L2")
    ax1.set_ylabel("L2")
    ax1.set_xlabel("Stochasticity")
    plt.savefig(f"losses_{'_'.join(group)}.png")
    plt.close()

    f = plt.figure(figsize=(5, 5))
    spec = f.add_gridspec(1, 2)
    ax1 = f.add_subplot(spec[0, :])
    colour_count = 0
    for member in group:
        with open(f"{member}/diversity_data.pk", "rb") as f:
            loss_data = pk.load(f)

        for variant, data in loss_data.items():
            sns.lineplot(data["stoch"], data["DIV"], estimator="mean", ci="sd", label=f"{member}-{variant}", ax=ax1, color=colours[colour_count])
            colour_count += 1
    ax1.set_title("Diversity Score")
    ax1.set_ylabel("Diversity")
    ax1.set_xlabel("Stochasticity")
    plt.savefig(f"diversity_{'_'.join(group)}.png")
    plt.close()

    f = plt.figure(figsize=(5, 5))
    spec = f.add_gridspec(1, 2)
    ax1 = f.add_subplot(spec[0, :])
    colour_count = 0
    for member in group:
        with open(f"{member}/pct_moved_data.pk", "rb") as f:
            loss_data = pk.load(f)

        for variant, data in loss_data.items():
            sns.lineplot(data["stoch"], data["PCT"], estimator="mean", ci="sd", label=f"{member}-{variant}", ax=ax1,
                         color=colours[colour_count])
            colour_count += 1
    ax1.set_title("% Solutions Moving The Ball")
    ax1.set_ylabel("%")
    ax1.set_xlabel("Stochasticity")
    plt.savefig(f"pct_moved_{'_'.join(group)}.png")
    plt.close()

    f = plt.figure(figsize=(5, 5))
    spec = f.add_gridspec(1, 2)
    ax1 = f.add_subplot(spec[0, :])
    colour_count = 0
    for member in group:
        with open(f"{member}/dist_data.pk", "rb") as f:
            loss_data = pk.load(f)

        for variant, data in loss_data.items():
            sns.lineplot(data["stoch"], data["MDE"], estimator="mean", ci="sd", label=f"{member}-{variant}", ax=ax1,
                         color=colours[colour_count])
            colour_count += 1
    ax1.set_title("Mean Distance Moved Excl. No-Move")
    ax1.set_ylabel("Distance")
    ax1.set_xlabel("Stochasticity")
    plt.savefig(f"dist_{'_'.join(group)}.png")
    plt.close()

    f = plt.figure(figsize=(5, 5))
    spec = f.add_gridspec(1, 2)
    ax1 = f.add_subplot(spec[0, :])
    colour_count = 0
    for member in group:
        with open(f"{member}/dist_data.pk", "rb") as f:
            loss_data = pk.load(f)

        for variant, data in loss_data.items():
            sns.lineplot(data["stoch"], data["VDE"], estimator="mean", ci="sd", label=f"{member}-{variant}", ax=ax1,
                         color=colours[colour_count])
            colour_count += 1
    ax1.set_title("Mean Variance of Distance Moved Excl. No-Move")
    ax1.set_ylabel("Variance")
    ax1.set_xlabel("Stochasticity")
    plt.savefig(f"dist_var_{'_'.join(group)}.png")
    plt.close()

    f = plt.figure(figsize=(5, 5))
    spec = f.add_gridspec(1, 2)
    ax1 = f.add_subplot(spec[0, :])
    colour_count = 0
    for member in group:
        with open(f"{member}/entropy_data.pk", "rb") as f:
            loss_data = pk.load(f)

        for variant, data in loss_data.items():
            sns.lineplot(data["stoch"], data["EVE"], estimator="mean", ci="sd", label=f"{member}-{variant}", ax=ax1,
                         color=colours[colour_count])
            colour_count += 1
    ax1.set_title("Entropy of Trajectory Positions Excl. No-Move")
    ax1.set_ylabel("Mean Entropy")
    ax1.set_xlabel("Stochasticity")
    plt.savefig(f"entropy_{'_'.join(group)}.png")
    plt.close()

    f = plt.figure(figsize=(5, 5))
    spec = f.add_gridspec(1, 2)
    ax1 = f.add_subplot(spec[0, :])
    colour_count = 0
    for member in group:
        with open(f"{member}/posvar_data.pk", "rb") as f:
            loss_data = pk.load(f)

        for variant, data in loss_data.items():
            sns.lineplot(data["stoch"], data["PV"], estimator="mean", ci="sd", label=f"{member}-{variant}", ax=ax1,
                         color=colours[colour_count])
            colour_count += 1
    ax1.set_title("Variance of Trajectory Positions")
    ax1.set_ylabel("Mean Variance")
    ax1.set_xlabel("Stochasticity")
    plt.savefig(f"posvar_{'_'.join(group)}.png")
    plt.close()

    break