import numpy as np
import seaborn as sns
import os
import matplotlib.pyplot as plt

def read_dpf(gen, path):
    with open(f"{path}/archive_{gen}.dat") as f:
        data = np.genfromtxt(f, delimiter=" ")
    dpf = data[:, -1]
    return dpf

def visualise_in_archive(gen, dpf):
    sns.distplot(dpf)
    plt.xlabel("Distance")
    plt.ylabel("Density")
    plt.title(f"Dpf in Archive - Gen {gen} - Total num. {len(dpf)}")
    plt.savefig(f"dpf{gen}.png")
    plt.close()

def visualise_in_exp(gen, path):
    pids = [pid for pid in os.listdir(path) if os.path.isdir(os.path.join(path, pid))]
    distances = []
    for pid in pids:
        distances.extend(list(read_dpf(gen, f"{path}/{pid}")))

    sns.distplot(distances)
    plt.xlabel("Distance")
    plt.ylabel("Density")
    plt.title(f"Dpf in Experiment - Gen {gen} - Total {len(distances)} - Mean {round(np.array(distances).mean(), 2)}")
    os.chdir(path)
    plt.savefig(f"dpf.png")
    plt.close()

def visualise_for_all_exp(gen, path):
    for ex in [exp for exp in os.listdir(path) if os.path.isdir(os.path.join(path, exp))]:
        visualise_in_exp(gen, f"{path}{ex}")

# dpf = read_dpf(6000, "/media/andwang1/SAMSUNG/MSC_INDIV/results_exp1/repeated_run1/results_balltrajectorysd_vae/gen6001_random1_fulllossfalse/2020-06-10_02_54_04_92957")
# visualise_in_archive(6000, dpf)

path = "/media/andwang1/SAMSUNG/MSC_INDIV/results_exp1/repeated_run1/results_balltrajectorysd_vae_beta0/"
visualise_for_all_exp(6000, path)
