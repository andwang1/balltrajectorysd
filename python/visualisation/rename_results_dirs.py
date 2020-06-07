import os
variant = "aurora"

os.chdir(f"/home/andwang1/airl/balltrajectorysd/results_exp1/repeated_run1/results_balltrajectorysd_{variant}")
dir_names = os.listdir()

for name in dir_names:
    if "--" not in name:
        continue

    components = name.split("_")
    args = [i.split("=")[-1] for i in components]
    new_name = f"gen{args[0]}_random{args[1]}_fullloss{args[2]}"

    os.rename(name, new_name)
