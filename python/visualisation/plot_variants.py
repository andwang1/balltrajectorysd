import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle as pk

path = "/media/andwang1/SAMSUNG/MSC_INDIV/results_box2d_bsd_exp1"
os.chdir(path)

plotting_groups = [
    ["l2"],
    ["l1", "smoothl1_longtrain"],
    ["l1", "l2"],
    ["extension03", "l2"],
    ["l1extension03", "l1"],
    ["l1beta0extension03", "l1beta0"],
    ["l2beta0", "l2"],
    ["l1beta0", "l1"],
    ["smoothl1_longtrain", "smoothl1_shorttrain"],
    ["l2nosample", "l2"],
    ["l1nosampletrain", "l1nosample", "l1"],
    ["l1beta0nosampletrain", "l1nosample", "l1"],
["l1beta0nosampletrain", "l1"],
    ["randomsolutions", "l1"],
    ["randomsolutions", "l1nosample"],
]

colours = ["blue", "brown", "grey", "red", "purple", "green", "pink"]

# make legend bigger
plt.rc('legend', fontsize=35)
# make lines thicker
plt.rc('lines', linewidth=4, linestyle='-.')
# make font bigger
plt.rc('font', size=30)
sns.set_style("dark")

for group in plotting_groups:
    print(f"Processing {group}")
    save_dir = f"plots/{'_'.join(group)}"
    os.makedirs(f"{save_dir}/pdf", exist_ok=True)
    f = plt.figure(figsize=(20, 20))
    spec = f.add_gridspec(1, 2)
    ax1 = f.add_subplot(spec[0, :])
    colour_count = 0
    for i, member in enumerate(group):
        with open(f"{member}/loss_data.pk", "rb") as f:
            log_data = pk.load(f)

        for variant, data in log_data.items():
            if "aurora" in variant:
                continue
            sns.lineplot(data["stoch"], data["AL"], estimator="mean", ci="sd", label=f"{member}-{variant}", ax=ax1, color=colours[colour_count])
            if i == 0 and len(group) > 1:
                ax1.lines[-1].set_linestyle("--")
            colour_count += 1
    ax1.set_title("Losses - Actual L2")
    ax1.set_ylabel("Mean L2")
    ax1.set_xlabel("Stochasticity")
    plt.savefig(f"{save_dir}/pdf/losses_{'_'.join(group)}.pdf")
    plt.savefig(f"{save_dir}/losses_{'_'.join(group)}.png")
    plt.close()

    f = plt.figure(figsize=(20, 20))
    spec = f.add_gridspec(1, 2)
    ax1 = f.add_subplot(spec[0, :])
    colour_count = 0
    for i, member in enumerate(group):
        with open(f"{member}/diversity_data.pk", "rb") as f:
            log_data = pk.load(f)

        for variant, data in log_data.items():
            sns.lineplot(data["stoch"], data["DIV"], estimator="mean", ci="sd", label=f"{member}-{variant}", ax=ax1, color=colours[colour_count])
            if i == 0 and len(group) > 1:
                ax1.lines[-1].set_linestyle("--")
            colour_count += 1
    ax1.set_title("Diversity Score")
    ax1.set_ylabel("Mean Diversity")
    ax1.set_xlabel("Stochasticity")
    plt.savefig(f"{save_dir}/pdf/diversity_{'_'.join(group)}.pdf")
    plt.savefig(f"{save_dir}/diversity_{'_'.join(group)}.png")
    plt.close()

    f = plt.figure(figsize=(20, 20))
    spec = f.add_gridspec(1, 2)
    ax1 = f.add_subplot(spec[0, :])
    colour_count = 0
    for i, member in enumerate(group):
        with open(f"{member}/pct_moved_data.pk", "rb") as f:
            log_data = pk.load(f)

        for variant, data in log_data.items():
            sns.lineplot(data["stoch"], data["PCT"], estimator="mean", ci="sd", label=f"{member}-{variant}", ax=ax1,
                         color=colours[colour_count])
            if i == 0 and len(group) > 1:
                ax1.lines[-1].set_linestyle("--")
            colour_count += 1
    ax1.set_title("% Solutions Moving The Ball")
    ax1.set_ylabel("Mean %")
    ax1.set_xlabel("Stochasticity")
    plt.savefig(f"{save_dir}/pdf/pct_moved_{'_'.join(group)}.pdf")
    plt.savefig(f"{save_dir}/pct_moved_{'_'.join(group)}.png")
    plt.close()

    f = plt.figure(figsize=(20, 20))
    spec = f.add_gridspec(1, 2)
    ax1 = f.add_subplot(spec[0, :])
    colour_count = 0
    for i, member in enumerate(group):
        with open(f"{member}/dist_data.pk", "rb") as f:
            log_data = pk.load(f)

        for variant, data in log_data.items():
            sns.lineplot(data["stoch"], data["MDE"], estimator="mean", ci="sd", label=f"{member}-{variant}", ax=ax1,
                         color=colours[colour_count])
            if i == 0 and len(group) > 1:
                ax1.lines[-1].set_linestyle("--")
            colour_count += 1
    ax1.set_title("Mean Distance Moved Excl. No-Move")
    ax1.set_ylabel("Mean Distance")
    ax1.set_xlabel("Stochasticity")
    plt.savefig(f"{save_dir}/pdf/dist_{'_'.join(group)}.pdf")
    plt.savefig(f"{save_dir}/dist_{'_'.join(group)}.png")
    plt.close()

    f = plt.figure(figsize=(20, 20))
    spec = f.add_gridspec(1, 2)
    ax1 = f.add_subplot(spec[0, :])
    colour_count = 0
    for i, member in enumerate(group):
        with open(f"{member}/dist_data.pk", "rb") as f:
            log_data = pk.load(f)

        for variant, data in log_data.items():
            sns.lineplot(data["stoch"], data["VDE"], estimator="mean", ci="sd", label=f"{member}-{variant}", ax=ax1,
                         color=colours[colour_count])
            if i == 0 and len(group) > 1:
                ax1.lines[-1].set_linestyle("--")
            colour_count += 1
    ax1.set_title("Mean Variance of Distance Moved Excl. No-Move")
    ax1.set_ylabel("Mean Variance")
    ax1.set_xlabel("Stochasticity")
    plt.savefig(f"{save_dir}/pdf/dist_var_{'_'.join(group)}.pdf")
    plt.savefig(f"{save_dir}/dist_var_{'_'.join(group)}.png")
    plt.close()

    f = plt.figure(figsize=(20, 20))
    spec = f.add_gridspec(1, 2)
    ax1 = f.add_subplot(spec[0, :])
    colour_count = 0
    for i, member in enumerate(group):
        with open(f"{member}/entropy_data.pk", "rb") as f:
            log_data = pk.load(f)

        for variant, data in log_data.items():
            sns.lineplot(data["stoch"], data["EVE"], estimator="mean", ci="sd", label=f"{member}-{variant}", ax=ax1,
                         color=colours[colour_count])
            if i == 0 and len(group) > 1:
                ax1.lines[-1].set_linestyle("--")
            colour_count += 1
    ax1.set_title("Entropy of Trajectory Positions Excl. No-Move")
    ax1.set_ylabel("Mean Mean Entropy")
    ax1.set_xlabel("Stochasticity")
    plt.savefig(f"{save_dir}/pdf/entropy_{'_'.join(group)}.pdf")
    plt.savefig(f"{save_dir}/entropy_{'_'.join(group)}.png")
    plt.close()

    f = plt.figure(figsize=(20, 20))
    spec = f.add_gridspec(1, 2)
    ax1 = f.add_subplot(spec[0, :])
    colour_count = 0
    for i, member in enumerate(group):
        with open(f"{member}/posvar_data.pk", "rb") as f:
            log_data = pk.load(f)

        for variant, data in log_data.items():
            sns.lineplot(data["stoch"], data["PV"], estimator="mean", ci="sd", label=f"{member}-{variant}", ax=ax1,
                         color=colours[colour_count])
            if i == 0 and len(group) > 1:
                ax1.lines[-1].set_linestyle("--")
            colour_count += 1
    ax1.set_title("Variance of Trajectory Positions")
    ax1.set_ylabel("Mean Mean Variance")
    ax1.set_xlabel("Stochasticity")
    plt.savefig(f"{save_dir}/pdf/posvar_{'_'.join(group)}.pdf")
    plt.savefig(f"{save_dir}/posvar_{'_'.join(group)}.png")
    plt.close()

    f = plt.figure(figsize=(20, 20))
    spec = f.add_gridspec(1, 2)
    ax1 = f.add_subplot(spec[0, :])
    colour_count = 0
    for i, member in enumerate(group):
        with open(f"{member}/recon_var_data.pk", "rb") as f:
            log_data = pk.load(f)

        for variant, data in log_data.items():
            if "aurora" in variant:
                continue
            sns.lineplot(data["stoch"], data["NMV"], estimator="mean", ci="sd", label=f"{member}-{variant}", ax=ax1,
                         color=colours[colour_count])
            if i == 0 and len(group) > 1:
                ax1.lines[-1].set_linestyle("--")
            colour_count += 1
    ax1.set_title("Reconstruction Var. of No-Move solutions")
    ax1.set_ylabel("Mean Mean Reconstruction Variance")
    ax1.set_xlabel("Stochasticity")
    plt.savefig(f"{save_dir}/pdf/recon_var_{'_'.join(group)}.pdf")
    plt.savefig(f"{save_dir}/recon_var_{'_'.join(group)}.png")
    plt.close()

    f = plt.figure(figsize=(20, 20))
    spec = f.add_gridspec(1, 2)
    ax1 = f.add_subplot(spec[0, :])
    colour_count = 0
    for i, member in enumerate(group):
        with open(f"{member}/latent_var_data.pk", "rb") as f:
            log_data = pk.load(f)

        for variant, data in log_data.items():
            if "aurora" in variant:
                continue
            sns.lineplot(data["stoch"], data["LV"], estimator="mean", ci="sd", label=f"{member}-{variant}", ax=ax1,
                         color=colours[colour_count])
            if i == 0 and len(group) > 1:
                ax1.lines[-1].set_linestyle("--")
            colour_count += 1
    ax1.set_ylabel("Mean Mean Variance")
    ax1.set_xlabel("Stochasticity")
    ax1.set_title(f"Variance of Latent Descriptors of No-Move Solutions")
    plt.savefig(f"{save_dir}/pdf/latent_var_{'_'.join(group)}.pdf")
    plt.savefig(f"{save_dir}/latent_var_{'_'.join(group)}.png")
    plt.close()

    # f = plt.figure(figsize=(20, 20))
    # spec = f.add_gridspec(1, 2)
    # ax1 = f.add_subplot(spec[0, :])
    # colour_count = 0
    # for i, member in enumerate(group):
    #     with open(f"{member}/loss_data.pk", "rb") as f:
    #         log_data = pk.load(f)
    #
    #     for variant, data in log_data.items():
    #         if "vae" not in variant:
    #             continue
    #         sns.lineplot(data["stoch"], data["ENVAR"], estimator="mean", ci="sd", label=f"{member}-{variant}", ax=ax1,
    #                      color=colours[colour_count])
    #         if i == 0 and len(group) > 1:
    #             ax1.lines[-1].set_linestyle("--")
    #         colour_count += 1
    # ax1.set_ylabel("Mean Mean Variance")
    # ax1.set_xlabel("Stochasticity")
    # ax1.set_title(f"Encoder Variance")
    # plt.savefig(f"{save_dir}/pdf/encoder_var_{'_'.join(group)}.pdf")
    # plt.savefig(f"{save_dir}/encoder_var_{'_'.join(group)}.png")
    # plt.close()