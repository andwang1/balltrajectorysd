import os
import shutil

search_path = "/media/andwang1/SAMSUNG/MSC_INDIV/results_exp1/repeated_run1/results_balltrajectorysd_vae"
dest_path = "/media/andwang1/SAMSUNG/MSC_INDIV/results_exp1/repeated_run1/results_balltrajectorysd_vae_beta0"
os.chdir(search_path)

experiments = os.listdir()

for experiment in experiments:
    os.chdir(f"{search_path}/{experiment}")
    pids = os.listdir()
    for pid in pids:
        os.chdir(f"{search_path}/{experiment}/{pid}")
        print(f"{search_path}/{experiment}/{pid}")
        with open("ae_loss.dat", "r") as f:
            try:
                beta = f.readlines()[0].split(",")[3].strip()
            except:
                beta = f.readlines()[3]

            if beta == "0":
                if not os.access(f"{dest_path}/{experiment}/", mode=os.F_OK):
                    os.makedirs(f"{dest_path}/{experiment}/")
                shutil.move(f"{search_path}/{experiment}/{pid}", f"{dest_path}/{experiment}/{pid}")


