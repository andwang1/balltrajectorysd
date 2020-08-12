import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle as pk
import numpy as np
import pandas as pd

path = "/media/andwang1/SAMSUNG/MSC_INDIV/results_box2d_bsd_exp1"
os.chdir(path)

plotting_groups = [
    ["tsne_nosampletrain", "tsne_nosampletrain_beta0"]
    # ["l2withsampling", "l2withsampling_beta0"],
    # ["l2", "l2withsampling"],
    # ["l2", "l2beta0"],
    # ["l2_nosampletrain", "l2"],
    # ["l2_nosampletrain", "l2beta0_nosampletrain"],
# ["tsne_nosampletrain_beta0", "sne_sampletrain", "sne_nosampletrain", "tsne_nosampletrain"],
# ["l1beta0nosample","l1nosample", "l1beta0"],
# ["l2beta0nosample","l2nosample", "l2beta0"],
#     ["sne_nosampletrain_beta0", "sne_nosampletrain"]
#     ["l2nosampletrain_add5", "l2nosampletrain"],
#     ["tsne_nosampletrain", "l2nosampletrain"],
# ["tsne_nosampletrain","tsne_nosampletrain_beta0"],
# ["sne_nosampletrain","sne_nosampletrain_beta0"],
    # ["tsne_nosampletrain", "sne_nosampletrain", "tsne_nosampletrain_beta0", "sne_nosampletrain_beta0"]
# ["l2nosampletrain", "sne_nosampletrain_beta0", "tsne_nosampletrain_beta0"]
# ["sne_nosampletrain_beta1", "tsne_nosampletrain_beta1", "l2beta1nosampletrain"],
# ["l2beta0nosampletrain", "l2beta1nosampletrain"],
# ["l2beta1nosampletrain", "l2"], #compare against ae
# ["l2nosample", "l2beta0nosample"]
# ["l1nosampletrain", "l2beta1nosampletrain"],
# ["l2beta1nosampletrain", "l2beta1nosample", "l2nosample", "l2beta0nosample"]
#     ["l2"],
#     ["l1", "smoothl1_longtrain"],
#     ["l1", "l2"],
#     ["extension03", "l2"],
#     ["l1extension03", "l1"],
#     ["l1beta0extension03", "l1beta0"],
#     ["l2beta0", "l2"],
#     ["l1beta0", "l1"],
#     ["smoothl1_longtrain", "smoothl1_shorttrain"],
#     ["l2nosample", "l2"],
#     ["l1nosampletrain", "l1nosample", "l1"],
#     ["l1beta0nosampletrain", "l1nosample", "l1"],
# ["l1beta0nosampletrain", "l1"],
#     ["randomsolutions", "l1"],
#     ["randomsolutions", "l1nosample"],
#     ["l2beta1nosampletrain", "l2"],
# ["l2beta1nosampletrain", "l2nosample"],
]

# f"{member}-{variant_name}" if not (member == "l2beta0nosampletrain" and variant_name == "vaefulllossfalse") else f"{member}-{variant_name} / l2-ae"

colours = ["blue", "brown", "grey", "green", "purple", "red", "pink", "orange"]

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
            if "aurora" in variant :#or variant.startswith("ae"):
                continue
            variant_name = variant if not variant.startswith("ae") else "ae"
            sns.lineplot(data["stoch"], data["AL"], estimator=np.median, ci=None, label=f"{member}-{variant_name}", ax=ax1, color=colours[colour_count])
            if i == 0 and len(group) > 1:
                ax1.lines[-1].set_linestyle("--")
            empty_keys = [k for k in data if not len(data[k])]
            for k in empty_keys:
                del data[k]
            data_stats = pd.DataFrame(data)[["stoch", "AL"]].groupby("stoch").describe()
            quart25 = data_stats[('AL', '25%')]
            quart75 = data_stats[('AL', '75%')]
            ax1.fill_between([0, 1, 2, 3, 4, 5], quart25, quart75, alpha=0.3, color=colours[colour_count])
            colour_count += 1
    ax1.set_title("Losses - Actual L2")
    ax1.set_ylabel("L2")
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
            if "aurora" in variant :#or variant.startswith("ae"):
                continue
            variant_name = variant if not variant.startswith("ae") else "ae"
            sns.lineplot(data["stoch"], data["DIV"], estimator=np.median, ci=None, label=f"{member}-{variant_name}", ax=ax1, color=colours[colour_count])
            data_stats = pd.DataFrame(data)[["stoch", "DIV"]].groupby("stoch").describe()
            quart25 = data_stats[('DIV', '25%')]
            quart75 = data_stats[('DIV', '75%')]
            ax1.fill_between([0, 1, 2, 3, 4, 5], quart25, quart75, alpha=0.3, color=colours[colour_count])
            if i == 0 and len(group) > 1:
                ax1.lines[-1].set_linestyle("--")
            colour_count += 1
    ax1.set_title("Diversity Score")
    ax1.set_ylabel("Diversity")
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
            if "aurora" in variant :#or variant.startswith("ae"):
                continue
            variant_name = variant if not variant.startswith("ae") else "ae"
            sns.lineplot(data["stoch"], data["PCT"], estimator=np.median, ci=None, label=f"{member}-{variant_name}", ax=ax1,
                         color=colours[colour_count])
            data_stats = pd.DataFrame(data)[["stoch", "PCT"]].groupby("stoch").describe()
            quart25 = data_stats[('PCT', '25%')]
            quart75 = data_stats[('PCT', '75%')]
            ax1.fill_between([0, 1, 2, 3, 4, 5], quart25, quart75, alpha=0.3, color=colours[colour_count])
            if i == 0 and len(group) > 1:
                ax1.lines[-1].set_linestyle("--")
            colour_count += 1
    ax1.set_title("% Solutions Moving The Ball")
    ax1.set_ylabel("%")
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
            if "aurora" in variant :#or variant.startswith("ae"):
                continue
            variant_name = variant if not variant.startswith("ae") else "ae"
            sns.lineplot(data["stoch"], data["MDE"], estimator=np.median, ci=None, label=f"{member}-{variant_name}", ax=ax1,
                         color=colours[colour_count])
            data_stats = pd.DataFrame(data)[["stoch", "MDE"]].groupby("stoch").describe()
            quart25 = data_stats[('MDE', '25%')]
            quart75 = data_stats[('MDE', '75%')]
            ax1.fill_between([0, 1, 2, 3, 4, 5], quart25, quart75, alpha=0.3, color=colours[colour_count])
            if i == 0 and len(group) > 1:
                ax1.lines[-1].set_linestyle("--")
            colour_count += 1
    ax1.set_title("Distance Moved Excl. No-Move")
    ax1.set_ylabel("Distance")
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
            if "aurora" in variant :#or variant.startswith("ae"):
                continue
            variant_name = variant if not variant.startswith("ae") else "ae"
            sns.lineplot(data["stoch"], data["VDE"], estimator=np.median, ci=None, label=f"{member}-{variant_name}", ax=ax1,
                         color=colours[colour_count])
            data_stats = pd.DataFrame(data)[["stoch", "VDE"]].groupby("stoch").describe()
            quart25 = data_stats[('VDE', '25%')]
            quart75 = data_stats[('VDE', '75%')]
            ax1.fill_between([0, 1, 2, 3, 4, 5], quart25, quart75, alpha=0.3, color=colours[colour_count])
            if i == 0 and len(group) > 1:
                ax1.lines[-1].set_linestyle("--")
            colour_count += 1
    ax1.set_title("Variance of Distance Moved Excl. No-Move")
    ax1.set_ylabel("Variance")
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
            if "aurora" in variant :#or variant.startswith("ae"):
                continue
            variant_name = variant if not variant.startswith("ae") else "ae"
            sns.lineplot(data["stoch"], data["EVE"], estimator=np.median, ci=None, label=f"{member}-{variant_name}", ax=ax1,
                         color=colours[colour_count])
            data_stats = pd.DataFrame(data)[["stoch", "EVE"]].groupby("stoch").describe()
            quart25 = data_stats[('EVE', '25%')]
            quart75 = data_stats[('EVE', '75%')]
            ax1.fill_between([0, 1, 2, 3, 4, 5], quart25, quart75, alpha=0.3, color=colours[colour_count])
            if i == 0 and len(group) > 1:
                ax1.lines[-1].set_linestyle("--")
            colour_count += 1
    ax1.set_title("Entropy of Trajectory Positions Excl. No-Move")
    ax1.set_ylabel("Entropy")
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
            if "aurora" in variant :#or variant.startswith("ae"):
                continue
            variant_name = variant if not variant.startswith("ae") else "ae"
            sns.lineplot(data["stoch"], data["PV"], estimator=np.median, ci=None, label=f"{member}-{variant_name}", ax=ax1,
                         color=colours[colour_count])
            if i == 0 and len(group) > 1:
                ax1.lines[-1].set_linestyle("--")
            data_stats = pd.DataFrame(data)[["stoch", "PV"]].groupby("stoch").describe()
            quart25 = data_stats[('PV', '25%')]
            quart75 = data_stats[('PV', '75%')]
            ax1.fill_between([0, 1, 2, 3, 4, 5], quart25, quart75, alpha=0.3, color=colours[colour_count])

            colour_count += 1
    ax1.set_title("Variance of Trajectory Positions")
    ax1.set_ylabel("Variance")
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
            if "aurora" in variant :#or variant.startswith("ae"):
                continue
            variant_name = variant if not variant.startswith("ae") else "ae"
            sns.lineplot(data["stoch"], data["NMV"], estimator=np.median, ci=None, label=f"{member}-{variant_name}", ax=ax1,
                         color=colours[colour_count])
            data_stats = pd.DataFrame(data)[["stoch", "NMV"]].groupby("stoch").describe()
            quart25 = data_stats[('NMV', '25%')]
            quart75 = data_stats[('NMV', '75%')]
            ax1.fill_between([0, 1, 2, 3, 4, 5], quart25, quart75, alpha=0.3, color=colours[colour_count])
            if i == 0 and len(group) > 1:
                ax1.lines[-1].set_linestyle("--")
            colour_count += 1
    ax1.set_title("Var. of Constructions of No-Move solutions")
    ax1.set_ylabel("Variance")
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
            if "aurora" in variant :#or variant.startswith("ae"):
                continue
            variant_name = variant if not variant.startswith("ae") else "ae"
            sns.lineplot(data["stoch"], data["LV"], estimator=np.median, ci=None, label=f"{member}-{variant_name}", ax=ax1,
                         color=colours[colour_count])
            data_stats = pd.DataFrame(data)[["stoch", "LV"]].groupby("stoch").describe()
            quart25 = data_stats[('LV', '25%')]
            quart75 = data_stats[('LV', '75%')]
            ax1.fill_between([0, 1, 2, 3, 4, 5], quart25, quart75, alpha=0.3, color=colours[colour_count])
            if i == 0 and len(group) > 1:
                ax1.lines[-1].set_linestyle("--")
            colour_count += 1
    ax1.set_ylabel("Variance")
    ax1.set_xlabel("Stochasticity")
    ax1.set_title(f"Variance of Latent Descriptors of No-Move Solutions")
    plt.savefig(f"{save_dir}/pdf/latent_var_{'_'.join(group)}.pdf")
    plt.savefig(f"{save_dir}/latent_var_{'_'.join(group)}.png")
    plt.close()



    f = plt.figure(figsize=(20, 20))
    spec = f.add_gridspec(1, 2)
    ax1 = f.add_subplot(spec[0, :])
    colour_count = 0
    for i, member in enumerate(group):
        with open(f"{member}/loss_data.pk", "rb") as f:
            log_data = pk.load(f)

        for variant, data in log_data.items():
            if "aurora" in variant or variant.startswith("ae") or "fulllossfalse" in variant:
                continue
            sns.lineplot(data["stoch"], data["TSNE"], estimator=np.median, ci=None, label=f"{member}-{variant}",
                         ax=ax1,
                         color=colours[colour_count])
            data = pd.DataFrame(data)[["stoch", "TSNE"]]
            data_stats = data.groupby("stoch").describe()
            quart25 = data_stats[('TSNE', '25%')]
            quart75 = data_stats[('TSNE', '75%')]
            ax1.fill_between([0, 1, 2, 3, 4, 5], quart25, quart75, alpha=0.3, color=colours[colour_count])
            if i == 0 and len(group) > 1:
                ax1.lines[-1].set_linestyle("--")
            colour_count += 1
    ax1.set_ylabel("SNE" if "tsne" not in member else "T-SNE")
    ax1.set_xlabel("Stochasticity")
    ax1.set_title("SNE" if "tsne" not in member else "T-SNE")
    plt.savefig(f"{save_dir}/pdf/tsne_{'_'.join(group)}.pdf")
    plt.savefig(f"{save_dir}/tsne_{'_'.join(group)}.png")
    plt.close()

    f = plt.figure(figsize=(20, 20))
    spec = f.add_gridspec(1, 2)
    ax1 = f.add_subplot(spec[0, :])
    colour_count = 0
    for i, member in enumerate(group):
        with open(f"{member}/loss_data.pk", "rb") as f:
            log_data = pk.load(f)

        for variant, data in log_data.items():
            if "aurora" in variant or variant.startswith("ae") or "fulllossfalse" in variant:
                continue
            sns.lineplot(data["stoch"], data["KL"], estimator=np.median, ci=None, label=f"{member}-{variant}",
                         ax=ax1,
                         color=colours[colour_count])
            data = pd.DataFrame(data)[["stoch", "KL"]]
            data_stats = data.groupby("stoch").describe()
            quart25 = data_stats[('KL', '25%')]
            quart75 = data_stats[('KL', '75%')]
            ax1.fill_between([0, 1, 2, 3, 4, 5], quart25, quart75, alpha=0.3, color=colours[colour_count])
            if i == 0 and len(group) > 1:
                ax1.lines[-1].set_linestyle("--")
            colour_count += 1
    ax1.set_ylabel("KL")
    ax1.set_xlabel("Stochasticity")
    ax1.set_title("KL Loss")
    plt.savefig(f"{save_dir}/pdf/kl_{'_'.join(group)}.pdf")
    plt.savefig(f"{save_dir}/kl_{'_'.join(group)}.png")
    plt.close()

    f = plt.figure(figsize=(20, 20))
    spec = f.add_gridspec(1, 2)
    ax1 = f.add_subplot(spec[0, :])
    colour_count = 0
    for i, member in enumerate(group):
        with open(f"{member}/loss_data.pk", "rb") as f:
            log_data = pk.load(f)

        for variant, data in log_data.items():
            if "aurora" in variant or variant.startswith("ae"):
                continue
            sns.lineplot(data["stoch"], data["ENVAR"] / 2, estimator=np.median, ci=None, label=f"{member}-{variant}",
                         ax=ax1,
                         color=colours[colour_count])
            data = pd.DataFrame(data)[["stoch", "ENVAR"]]
            data["ENVAR"] /= 2
            data_stats = data.groupby("stoch").describe()
            quart25 = data_stats[('ENVAR', '25%')]
            quart75 = data_stats[('ENVAR', '75%')]
            ax1.fill_between([0, 1, 2, 3, 4, 5], quart25, quart75, alpha=0.3, color=colours[colour_count])
            if i == 0 and len(group) > 1:
                ax1.lines[-1].set_linestyle("--")
            colour_count += 1
    ax1.set_ylabel("Variance")
    ax1.set_xlabel("Stochasticity")
    ax1.set_title(f"Encoder Variance")
    plt.savefig(f"{save_dir}/pdf/encoder_var_{'_'.join(group)}.pdf")
    plt.savefig(f"{save_dir}/encoder_var_{'_'.join(group)}.png")
    plt.close()