import os
import sys
from diversity import plot_diversity_in_dir
from dist_grid import plot_dist_grid_in_dir
from pos_var_grid import plot_pos_var_grid_in_dir
from entropy_grid import plot_entropy_grid_in_dir
from ae_loss_AE import plot_loss_in_dir_AE
from ae_loss_VAE import plot_loss_in_dir_VAE
from latent_space import plot_latent_space_in_dir

current_path = os.getcwd()
sys.path.append("/git/sferes2/exp/balltrajectorysd/python/visualisation")

GENERATE_EACH_IMAGE = True

path = sys.argv[1]
application = sys.argv[2]
variant = application.split("_")[-1]
is_full_loss = "true" in sys.argv[3]

print(f"PROCESSING VISUALISATIONS - {path}")
plot_latent_space_in_dir(path)
div_dict, max_diversity = plot_diversity_in_dir(path, GENERATE_EACH_IMAGE)
plot_dist_grid_in_dir(path, GENERATE_EACH_IMAGE)
plot_pos_var_grid_in_dir(path, GENERATE_EACH_IMAGE)
plot_entropy_grid_in_dir(path, GENERATE_EACH_IMAGE)
# PID level plotting
if variant == "vae":
    plot_loss_in_dir_VAE(path, is_full_loss, GENERATE_EACH_IMAGE, plot_total_L2=False)
elif variant == "aurora":
    plot_loss_in_dir_AE(path, GENERATE_EACH_IMAGE, is_aurora=True)
else:
    plot_loss_in_dir_AE(path, GENERATE_EACH_IMAGE)

os.chdir(current_path)