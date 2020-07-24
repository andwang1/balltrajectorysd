//
// Created by Luca Grillotti on 21/05/2020.
//

//| This file is a part of the sferes2 framework.
//| Copyright 2016, ISIR / Universite Pierre et Marie Curie (UPMC)
//| Main contributor(s): Jean-Baptiste Mouret, mouret@isir.fr
//|
//| This software is a computer program whose purpose is to facilitate
//| experiments in evolutionary computation and evolutionary robotics.
//|
//| This software is governed by the CeCILL license under French law
//| and abiding by the rules of distribution of free software.  You
//| can use, modify and/ or redistribute the software under the terms
//| of the CeCILL license as circulated by CEA, CNRS and INRIA at the
//| following URL "http://www.cecill.info".
//|
//| As a counterpart to the access to the source code and rights to
//| copy, modify and redistribute granted by the license, users are
//| provided only with a limited warranty and the software's author,
//| the holder of the economic rights, and the successive licensors
//| have only limited liability.
//|
//| In this respect, the user's attention is drawn to the risks
//| associated with loading, using, modifying and/or developing or
//| reproducing the software by the user in light of its specific
//| status of free software, that may mean that it is complicated to
//| manipulate, and that also therefore means that it is reserved for
//| developers and experienced professionals having in-depth computer
//| knowledge. Users are therefore encouraged to load and test the
//| software's suitability as regards their requirements in conditions
//| enabling the security of their systems and/or data to be ensured
//| and, more generally, to use and operate it in the same conditions
//| as regards security.
//|
//| The fact that you are presently reading this means that you have
//| had knowledge of the CeCILL license and that you accept its terms.

#ifdef AURORA
#define AE
#endif

#include <iostream>
#include <algorithm>
#include <unistd.h>

#include <boost/program_options.hpp>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include <torch/torch.h>
#include <torch/script.h> // One-stop header.

#include <sferes/eval/parallel.hpp>
#include <sferes/gen/evo_float.hpp>
#include <sferes/run.hpp>
#include <sferes/fit/fit_qd.hpp>
#include <sferes/qd/container/archive.hpp>
#include <sferes/qd/container/kdtree_storage.hpp>
#include <sferes/qd/quality_diversity.hpp>
#include <sferes/qd/selector/value_selector.hpp>
#include <sferes/qd/selector/score_proportionate.hpp>
#include <sferes/qd/selector/noselection.hpp>

#include <sferes/stat/best_fit.hpp>
#include <sferes/stat/qd_container.hpp>
#include <sferes/stat/qd_selection.hpp>
#include <sferes/stat/qd_progress.hpp>

#include "stat/stat_current_gen.hpp"
#include "stat/stat_model_autoencoder.hpp"
#include "stat/stat_modifier.hpp"
#include "stat/stat_traj.hpp"
#include "stat/stat_ae_losses.hpp"
#include "stat/stat_diversity.hpp"
#include "stat/stat_distances.hpp"
#include "stat/stat_similarity.hpp"
#include "stat/stat_entropy.hpp"
#include "stat/stat_notmoved_recon_var.hpp"

#include "modifier/dimensionality_reduction.hpp"

#ifdef AURORA
#include "modifier/network_loader_pytorch_AURORA.hpp"
#else
#include "modifier/network_loader_pytorch.hpp"
#endif

#include "params.hpp"
#include "trajectory.hpp"
#include "phen.hpp"
#include "archive_2.hpp"
#include "quality_diversity_2.hpp"

struct Arguments {
    size_t number_cpus;
    double pct_random;
    bool full_loss;
    bool l2_loss;
    size_t number_gen;
    size_t beta;
    double pct_extension;
    unsigned int loss_func;
    bool sample;
    bool tsne;
};

void get_arguments(const boost::program_options::options_description &desc, Arguments &arg, int argc, char **argv) {
    // For the moment, only returning number of threads
    boost::program_options::variables_map vm;
    boost::program_options::parsed_options parsed = boost::program_options::command_line_parser(argc, argv).options(
            desc).allow_unregistered().run();

    boost::program_options::store(parsed, vm);
    boost::program_options::notify(vm);
    arg.number_cpus = vm["number-cpus"].as<size_t>();
    arg.pct_random = vm["pct-random"].as<double>();
    arg.full_loss = vm["full-loss"].as<bool>();
    arg.number_gen = vm["number-gen"].as<size_t>();
    arg.beta = vm["beta"].as<size_t>();
    arg.pct_extension = vm["pct-extension"].as<double>();
    arg.loss_func = vm["loss-func"].as<unsigned int>();
    arg.sample = vm["sample"].as<bool>();
    arg.tsne = vm["tsne"].as<bool>();
}

int main(int argc, char **argv) {
    // threading options
    boost::program_options::options_description desc;
    Arguments arg{};

    desc.add_options()
                ("number-gen", boost::program_options::value<size_t>(), "Set Number of Generations");
    desc.add_options()
                ("number-cpus", boost::program_options::value<size_t>(), "Set Number of CPUs");
    desc.add_options()
                ("pct-random", boost::program_options::value<double>(), "Set Pct of random trajectories");
    desc.add_options()
                ("full-loss", boost::program_options::value<bool>(), "Full VAE loss or just L2");
    desc.add_options()
                ("beta", boost::program_options::value<size_t>(), "Beta Coefficient");
    desc.add_options()
                ("pct-extension", boost::program_options::value<double>(), "% of Phenotypes to regenerate for training");
    desc.add_options()
                ("loss-func", boost::program_options::value<unsigned int>(), "Loss function: 0 = SqRoot, 1 = L1, 2 = L2");
    desc.add_options()
                ("sample", boost::program_options::value<bool>(), "Sample Encoder for BD");
    desc.add_options()
                ("tsne", boost::program_options::value<bool>(), "True for TSNE, False for");

    get_arguments(desc, arg, argc, argv);

    srand(time(0));

    // threading tool
    tbb::task_scheduler_init init(arg.number_cpus);

    typedef Params params_t;
    Params::nov::l = 0;

    // cmd line arguments
    // number of generations
    Params::pop::nb_gen = arg.number_gen;
    // pct of random trajectories in population
    Params::random::pct_random = arg.pct_random;
    // VAE loss (full_loss) or L2 loss
    Params::ae::full_loss = arg.full_loss;
    // KL Beta
    Params::ae::beta = arg.beta;
    // Additional phenotypes to retrain on
    Params::ae::pct_extension = arg.pct_extension;
    // loss function
    Params::ae::loss_function = static_cast<Params::ae::loss>(arg.loss_func);
    // sample encoder for BD
    Params::qd::sample = arg.sample;
    // TSNE or SNE
    Params::ae::TSNE = arg.tsne;

    typedef Trajectory<params_t> fit_t;
    typedef sferes::gen::EvoFloat<Params::qd::gen_dim, params_t> gen_t;
    typedef sferes::phen::Custom_Phen<gen_t, fit_t, params_t> phen_t;

    typedef NetworkLoaderAutoEncoder<params_t> network_loader_t;
    typedef sferes::modif::DimensionalityReduction<phen_t, params_t, network_loader_t> modifier_t;

    // For the Archive, you can chose one of the following storage:
    // kD_tree storage, recommended for small behavioral descriptors (behav_dim<10)
    typedef  std::conditional<
                    params_t::qd::behav_dim <= 10,
                    sferes::qd::container::KdtreeStorage<boost::shared_ptr<phen_t>, params_t::qd::behav_dim>,
                    sferes::qd::container::SortBasedStorage< boost::shared_ptr<phen_t>>
                >::type storage_t;

    typedef Archive_2<phen_t, storage_t, params_t> container_t;
    // typedef sferes::qd::container::Archive<phen_t, storage_t, params_t> container_t;

    // if GRAPHICS
    // typedef sferes::eval::Eval<Params> eval_t;
    typedef sferes::eval::Parallel<params_t> eval_t;

    typedef boost::fusion::vector<
                    sferes::stat::QdContainer<phen_t, params_t>,
                    sferes::stat::QdProgress<phen_t, params_t>,
                    sferes::stat::Losses<phen_t, params_t>,
                    sferes::stat::Trajectories<phen_t, params_t>,
                    sferes::stat::Diversity<phen_t, params_t>,
                    sferes::stat::Distances<phen_t, params_t>,
                    // similarity needs to run after distances as it gets info on whether the ball moved from distances
                    sferes::stat::Similarity<phen_t, params_t>,
                    sferes::stat::Entropy<phen_t, params_t>,
                    sferes::stat::Modifier<phen_t, params_t>,
                    sferes::stat::NotMovedReconVar<phen_t, params_t>
                > stat_t;


    typedef sferes::qd::selector::Uniform<phen_t, params_t> selector_t;

    typedef QualityDiversity_2 <phen_t, eval_t, stat_t, modifier_t, selector_t, container_t, params_t> ea_t;

    ea_t ea;

    sferes::run_ea(argc, argv, ea, desc);

    return 0;
}

