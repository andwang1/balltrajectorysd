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
#include <sferes/modif/dummy.hpp>
#include <sferes/phen/parameters.hpp>
#include <sferes/run.hpp>
#include <sferes/stat/best_fit.hpp>
#include <sferes/stat/qd_container.hpp>
#include <sferes/stat/qd_selection.hpp>
#include <sferes/stat/qd_progress.hpp>

#include <sferes/fit/fit_qd.hpp>
#include <sferes/qd/container/archive.hpp>
#include <sferes/qd/container/kdtree_storage.hpp>
#include <sferes/qd/quality_diversity.hpp>
#include <sferes/qd/selector/value_selector.hpp>
#include <sferes/qd/selector/score_proportionate.hpp>

#include "modifier/network_loader_pytorch.hpp"
#include "modifier/dimensionality_reduction.hpp"

#include "stat/utils.hpp"
#include "stat/stat_projection.hpp"
#include "stat/stat_images_observations.hpp"
#include "stat/stat_images_reconstructions_obs.hpp"
#include "stat/stat_model_autoencoder.hpp"
#include "stat/stat_modifier.hpp"

#include "fastsim_display.hpp"
#include "fit_maze.hpp"
#include "params_maze.hpp"

// quick hack to have "write" access to the container, this need to be added to the main API later.
template<typename Phen, typename Eval, typename Stat, typename FitModifier, typename Select, typename Container, typename Params, typename Exact = stc::Itself>
class QualityDiversity_2
        : public sferes::qd::QualityDiversity<Phen, Eval, Stat, FitModifier, Select, Container, Params, typename stc::FindExact<QualityDiversity_2<Phen, Eval, Stat, FitModifier, Select, Container, Params, Exact>, Exact>::ret> {

public:

    typedef Phen phen_t;
    typedef boost::shared_ptr <Phen> indiv_t;
    typedef typename std::vector<indiv_t> pop_t;
    typedef typename pop_t::iterator it_t;

    pop_t pop_advers;
    // pop_t& get_pop_advers() { return this->pop_advers; }

    Container &container() { return this->_container; }


    void add(pop_t &pop_off, std::vector<bool> &added, pop_t &pop_parents) {
        this->_add(pop_off, added, pop_parents);
    }

    // Same function, but without the need of parent.
    void add(pop_t &pop_off, std::vector<bool> &added) {
        std::cout << "adding with l: " << Params::nov::l << std::endl;
        this->_add(pop_off, added);
    }

};

struct Arguments {
    size_t number_threads;
};

void get_arguments(const boost::program_options::options_description &desc, Arguments &arg, int argc, char **argv) {
    // For the moment, only returning number of threads
    boost::program_options::variables_map vm;
    boost::program_options::parsed_options parsed = boost::program_options::command_line_parser(argc, argv).options(
            desc).allow_unregistered().run();

    boost::program_options::store(parsed, vm);
    boost::program_options::notify(vm);
    arg.number_threads = vm["number-threads"].as<size_t>();
}

int main(int argc, char **argv) {
    boost::program_options::options_description desc;
    Arguments arg{};

    desc.add_options()
                ("number-threads", boost::program_options::value<size_t>(), "Set Number of Threads");

    get_arguments(desc, arg, argc, argv);

    srand(time(0));

    tbb::task_scheduler_init init(arg.number_threads);

    init_fastsim_settings();

    typedef ParamsMaze params_t;
    ParamsMaze::nov::l = 0;

    typedef sferes::phen::Parameters<sferes::gen::EvoFloat<1, param_t>, sferes::fit::FitDummy<>, param_t> weight_t;
    typedef sferes::phen::Parameters<sferes::gen::EvoFloat<1, param_t>, sferes::fit::FitDummy<>, param_t> bias_t;
    typedef nn::PfWSum<weight_t> pf_t;
    typedef nn::AfTanh<bias_t> af_t;
    typedef nn::Neuron<pf_t, af_t> neuron_t;
    typedef nn::Connection<weight_t> connection_t;

    typedef sferes::gen::GenMlp<neuron_t, connection_t, param_t> gen_t;
    typedef HardMaze<param_t> fit_t;
    typedef sferes::phen::Dnn<gen_t, fit_t, param_t> phen_t;

    typedef NetworkLoaderAutoEncoder<params_t> network_loader_t;
    typedef sferes::modif::DimensionalityReduction<phen_t, param_t, network_loader_t> modifier_t;

    // For the Archive, you can chose one of the following storage:
    // kD_tree storage, recommended for small behavioral descriptors (behav_dim<10)
    typedef  std::conditional<
                    param_t::qd::behav_dim <= 10,
                    sferes::qd::container::KdtreeStorage<boost::shared_ptr <phen_t>, param_t::qd::behav_dim >,
                    sferes::qd::container::SortBasedStorage< boost::shared_ptr<phen_t>>
                >::type storage_t;

    typedef sferes::qd::container::Archive<phen_t, storage_t, param_t> container_t;

    typedef sferes::eval::Parallel<param_t> eval_t;
    // typedef eval::Eval<Params> eval_t;

    typedef boost::fusion::vector<
                    sferes::stat::CurrentGen<phen_t, param_t>,
                    sferes::stat::QdContainer<phen_t, param_t>,
                    sferes::stat::QdProgress<phen_t, param_t>,
                    sferes::stat::Projection<phen_t, param_t>,
                    sferes::stat::ImagesObservations<phen_t, param_t>,
                    sferes::stat::ImagesReconstructionObs<phen_t, param_t>,
                    sferes::stat::ModelAutoencoder<phen_t, param_t>,
                    sferes::stat::Modifier<phen_t, param_t>
                > stat_t;

    typedef sferes::qd::selector::Uniform<phen_t, param_t> selector_t;

    typedef QualityDiversity_2 <phen_t, eval_t, stat_t, modifier_t, selector_t, container_t, param_t> ea_t;

    ea_t ea;

    sferes::run_ea(argc, argv, ea, desc);

    return 0;
}

