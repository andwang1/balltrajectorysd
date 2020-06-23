import os
variant = "aurora"

BASE_NAME = "/home/andwang1/airl/balltrajectorysd/results_box2d_exp1/first_run"

os.chdir(f"{BASE_NAME}/results_balltrajectorysd_{variant}")
dir_names = os.listdir()

for name in dir_names:
    if "--" not in name:
        continue

    components = name.split("_")
    args = [i.split("=")[-1] for i in components]
    # new_name = f"gen{args[0]}_random{args[1]}_fullloss{args[2]}"
    new_name = f"gen{args[0]}_random{args[1]}_fullloss{args[2]}_beta{args[3]}_extension{args[4]}_l2{args[5]}"

    os.rename(name, new_name)
