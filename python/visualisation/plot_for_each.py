import matplotlib.pyplot as plt
import os
import numpy as np
import pickle as pk
import seaborn as sns
from collections import defaultdict
from diversity import plot_diversity_in_dir
from pheno_circle import plot_pheno_in_dir
from ae_loss_AE import plot_loss_in_dir_AE
from ae_loss_VAE import plot_loss_in_dir_VAE

GENERATE_EACH_IMAGE = False
PLOT_TOTAL_L2 = False
START_GEN_LOSS_PLOT = 500

EXP_FOLDER = "/home/andwang1/airl/balltrajectorysd/results_exp1/test"
BASE_NAME = "results_balltrajectorysd_"
variants = [exp_name.split("_")[-1] for exp_name in os.listdir(EXP_FOLDER) if
            os.path.isdir(os.path.join(EXP_FOLDER, exp_name))]

# store all data
diversity_dict = defaultdict(list)
loss_dict = defaultdict(list)

for variant in variants:
    os.chdir(f"{EXP_FOLDER}/{BASE_NAME}{variant}")
    exp_names = [exp_name for exp_name in os.listdir() if
                 os.path.isdir(os.path.join(f"{EXP_FOLDER}/{BASE_NAME}{variant}", exp_name))]

    is_full_loss = [False] * len(exp_names)

    if variant == "vae":
        for i, name in enumerate(exp_names):
            if "true" in name:
                is_full_loss[i] = True

    variant_diversity_dict = defaultdict(list)
    variant_loss_dict = defaultdict(list)

    for i, exp in enumerate(exp_names):
        exp_path = f"{EXP_FOLDER}/{BASE_NAME}{variant}/{exp}"
        pids = [pid for pid in os.listdir(exp_path) if os.path.isdir(os.path.join(exp_path, pid))]
        for pid in pids:
            full_path = f"{EXP_FOLDER}/{BASE_NAME}{variant}/{exp}/{pid}"
            print(f"PROCESSING - {full_path}")
            div_dict, max_diversity = plot_diversity_in_dir(full_path, GENERATE_EACH_IMAGE)
            variant_diversity_dict[exp].append(div_dict)
            # PID level plotting
            if GENERATE_EACH_IMAGE:
                plot_pheno_in_dir(full_path)
            if variant == "vae":
                variant_loss_dict[exp].append(plot_loss_in_dir_VAE(full_path, is_full_loss[i], GENERATE_EACH_IMAGE, PLOT_TOTAL_L2))
            elif variant == "aurora":
                variant_loss_dict[exp].append(plot_loss_in_dir_AE(full_path, GENERATE_EACH_IMAGE, is_aurora=True))
            else:
                variant_loss_dict[exp].append(plot_loss_in_dir_AE(full_path, GENERATE_EACH_IMAGE))

        # experiment level plotting
        os.chdir(f"{EXP_FOLDER}/{BASE_NAME}{variant}/{exp}")

        # at experiment level, plot mean and stddev curves for diversity over generations
        x = list(variant_diversity_dict[exp][0].keys()) * len(variant_diversity_dict[exp])
        y = np.array([list(repetition.values()) for repetition in variant_diversity_dict[exp]])
        y = y.flatten()

        sns.lineplot(x, y, estimator="mean", ci="sd", label="Diversity")
        plt.title("Diversity Mean and Std.Dev.")
        plt.xlabel("Generation")
        plt.ylabel("Diversity")
        plt.hlines(max_diversity, 0, x[-1], linestyles="--", label="Max Diversity")
        plt.legend()
        plt.savefig("diversity.png")
        plt.close()

        if variant == "aurora":
            continue

        # at experiment level, plot losses
        L2_values = np.array([repetition["L2"][START_GEN_LOSS_PLOT:] for repetition in variant_loss_dict[exp]])
        AL_values = np.array([repetition["AL"][START_GEN_LOSS_PLOT:] for repetition in variant_loss_dict[exp]]).flatten()
        x = list(range(START_GEN_LOSS_PLOT, len(L2_values[0]) + START_GEN_LOSS_PLOT)) * len(L2_values)

        if "fulllosstrue" in exp:
            f = plt.figure(figsize=(10, 5))
            spec = f.add_gridspec(2, 2)
            ax1 = f.add_subplot(spec[0, :])
            ax2 = f.add_subplot(spec[1, :])
            VAR_values = np.array([repetition["VAR"][START_GEN_LOSS_PLOT:] for repetition in variant_loss_dict[exp]]).flatten()
            TL_values = np.array([repetition["TL"][START_GEN_LOSS_PLOT:] for repetition in variant_loss_dict[exp]]).flatten()
            KL_values = np.array([repetition["KL"][START_GEN_LOSS_PLOT:] for repetition in variant_loss_dict[exp]]).flatten()

            var_ax = ax1.twinx()
            ln3 = sns.lineplot(x, VAR_values, estimator="mean", ci="sd", label="Variance", ax=var_ax, color="green")
            var_ax.set_ylabel("Variance")

            ln4 = sns.lineplot(x, TL_values, estimator="mean", ci="sd", label="Total Loss", ax=ax2, color="red")
            ax2.set_ylabel("Total Loss")

            KL_ax = ax2.twinx()
            ln5 = sns.lineplot(x, KL_values, estimator="mean", ci="sd", label="KL", ax=KL_ax, color="blue")
            KL_ax.set_ylabel("KL")

            # first remove default legends automatically added then add combined set
            ax2.get_legend().remove()
            KL_ax.get_legend().remove()
            lns = ln4.get_lines() + ln5.get_lines()
            labs = [l.get_label() for l in lns]
            ax2.legend(lns, labs, loc='best')

        else:
            f = plt.figure(figsize=(5, 5))
            spec = f.add_gridspec(1, 2)
            ax1 = f.add_subplot(spec[0, :])

        # plot overall L2 and actual L2
        if PLOT_TOTAL_L2 or variant != "vae":
            ln1 = sns.lineplot(x, L2_values.flatten(), estimator="mean", ci="sd", label="Total L2", ax=ax1, color="red")
        ln2 = sns.lineplot(x, AL_values, estimator="mean", ci="sd", label="Actual L2", ax=ax1, color="blue")
        ax1.set_ylabel("L2")

        # add in legends, one return value of lineplot will have all lines on the axis
        ax1.get_legend().remove()
        if "fulllosstrue" in exp:
            var_ax.get_legend().remove()

        lns = ln2.get_lines() + ln3.get_lines() if "fulllosstrue" in exp else ln2.get_lines()
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc='best')

        ax1.set_title(f"Losses")
        if "fulllosstrue" in exp:
            ax2.set_xlabel("Generation")
        else:
            ax1.set_xlabel("Generation")

        plt.savefig("losses.png")
        plt.close()

    # variant plotting
    # retrieve stochasticity levels from file names
    os.chdir(f"{EXP_FOLDER}/{BASE_NAME}{variant}")
    generations = list(variant_diversity_dict[exp][0].keys())
    all_experiments = variant_diversity_dict.keys()
    stochasticities = []

    for name in all_experiments:
        components = name.split("_")
        # "random" part of experiment name
        stochasticities.append((components[1][len("random"):]))

    # remove duplicates from fulllosstrue and fulllossfalse and sort
    stochasticities = sorted(list(set(stochasticities)))

    # plot diversity across stochasticity for each generation
    for loss_type in ["fulllosstrue", "fulllossfalse"]:
        if variant != "vae" and loss_type == "fulllosstrue":
            continue
        for generation in generations:
            diversity_values = []
            stochasticity_values = []

            for stochasticity in stochasticities:
                # take correct dictionary according to stochasticity
                components[1] = f"random{stochasticity}"
                components[2] = loss_type
                for repetition in variant_diversity_dict["_".join(components)]:
                    diversity_values.append(repetition[generation])
                    stochasticity_values.append(stochasticity)

            sns.lineplot(stochasticity_values, diversity_values, estimator="mean", ci="sd", label="Diversity")
            plt.title(f"Diversity Mean and Std.Dev. - Gen {generation}")
            plt.xlabel("Stochasticity")
            plt.ylabel("Diversity")
            plt.hlines(max_diversity, 0, 1, linestyles="--", label="Max Diversity")
            plt.legend()
            if loss_type == "fulllosstrue":
                plt.savefig(f"diversity_gen{generation}_fullloss.png")
            else:
                plt.savefig(f"diversity_gen{generation}.png")
            plt.close()

    if variant == "aurora":
        continue

    # plot losses across stochasticity for each generation
    for loss_type in ["fulllosstrue", "fulllossfalse"]:
        if variant != "vae" and loss_type == "fulllosstrue":
            continue
        components[2] = loss_type
        L2_values = []
        AL_values = []
        VAR_values = []
        TL_values = []
        KL_values = []

        stochasticity_values = []

        is_data_recorded = True
        for stochasticity in stochasticities:
            # take correct dictionary according to stochasticity
            components[1] = f"random{stochasticity}"

            # if we do not have the data at the moment, skip plotting
            if not "_".join(components) in variant_loss_dict:
                is_data_recorded = False
                continue
            for repetition in variant_loss_dict["_".join(components)]:
                L2_values.append(repetition["L2"][START_GEN_LOSS_PLOT:])
                AL_values.append(repetition["AL"][START_GEN_LOSS_PLOT:])
                stochasticity_values.append([stochasticity] * len(repetition["AL"][START_GEN_LOSS_PLOT:]))

                if loss_type == "fulllosstrue":
                    VAR_values.append(repetition["VAR"][START_GEN_LOSS_PLOT:])
                    KL_values.append(repetition["KL"][START_GEN_LOSS_PLOT:])
                    TL_values.append(repetition["TL"][START_GEN_LOSS_PLOT:])

        if not is_data_recorded:
            continue

        # TR_EPOCHS = repetition["TR_EPOCHS"]

        # at experiment level, plot losses
        stochasticity_values = np.array(stochasticity_values).flatten()
        L2_values = np.array(L2_values).flatten()
        AL_values = np.array(AL_values).flatten()

        if loss_type == "fulllosstrue":
            VAR_values = np.array(VAR_values).flatten()
            KL_values = np.array(KL_values).flatten()
            TL_values = np.array(TL_values).flatten()

            f = plt.figure(figsize=(10, 5))
            spec = f.add_gridspec(2, 2)
            ax1 = f.add_subplot(spec[0, :])
            ax2 = f.add_subplot(spec[1, :])

            var_ax = ax1.twinx()
            ln3 = sns.lineplot(stochasticity_values, VAR_values, estimator="mean", ci="sd", label="Variance", ax=var_ax,
                               color="green")
            var_ax.set_ylabel("Variance")

            ln4 = sns.lineplot(stochasticity_values, TL_values, estimator="mean", ci="sd", label="Total Loss", ax=ax2, color="red")
            ax2.set_ylabel("Total Loss")

            KL_ax = ax2.twinx()
            ln5 = sns.lineplot(stochasticity_values, KL_values, estimator="mean", ci="sd", label="KL", ax=KL_ax, color="blue")
            KL_ax.set_ylabel("KL")

            # first remove default legends automatically added then add combined set
            ax2.get_legend().remove()
            KL_ax.get_legend().remove()
            lns = ln4.get_lines() + ln5.get_lines()
            labs = [l.get_label() for l in lns]
            ax2.legend(lns, labs, loc='best')

        else:
            f = plt.figure(figsize=(5, 5))
            spec = f.add_gridspec(1, 2)
            ax1 = f.add_subplot(spec[0, :])

        # plot overall L2 and actual L2
        if PLOT_TOTAL_L2 or variant != "vae":
            ln1 = sns.lineplot(stochasticity_values, L2_values, estimator="mean", ci="sd", label="Total L2", ax=ax1, color="red")
        ln2 = sns.lineplot(stochasticity_values, AL_values, estimator="mean", ci="sd", label="Actual L2", ax=ax1, color="blue")
        ax1.set_ylabel("L2")

        # add in legends, one return value of lineplot will have all lines on the axis
        ax1.get_legend().remove()
        if loss_type == "fulllosstrue":
            var_ax.get_legend().remove()

        lns = ln2.get_lines() + ln3.get_lines() if loss_type == "fulllosstrue" else ln2.get_lines()
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc='best')

        ax1.set_title(f"Losses")
        if loss_type == "fulllosstrue":
            ax2.set_xlabel("Stochasticity")
        else:
            ax1.set_xlabel("Stochasticity")

        if loss_type == "fulllosstrue":
            plt.savefig(f"losses_fullloss.png")
        else:
            plt.savefig(f"losses.png")

        plt.close()

    diversity_dict[variant].append(variant_diversity_dict)
    loss_dict[variant].append(variant_loss_dict)

os.chdir(f"{EXP_FOLDER}")

with open("diversity_data.pk", "wb") as f:
    pk.dump(diversity_dict, f)
with open("loss_data.pk", "wb") as f:
    pk.dump(loss_dict, f)
