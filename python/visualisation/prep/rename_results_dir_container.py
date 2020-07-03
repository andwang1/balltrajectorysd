import os
import sys

full_path = sys.argv[1]
full_path = full_path.split("/")
result_dir = full_path[-2]

path_to_variant = "/".join(full_path[:-2])
os.chdir(path_to_variant)

components = result_dir.split("_")
args = [i.split("=")[-1] for i in components]

new_result_dir_name = f"gen{args[0]}_random{args[1]}_fullloss{args[2]}_beta{args[3]}_extension{args[4]}_lossfunc{args[5]}_sample{args[6]}"

os.rename(result_dir, new_result_dir_name)
