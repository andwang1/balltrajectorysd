//
// Created by Luca Grillotti on 29/04/2020.
//

#ifndef EXAMPLE_PYTORCH_PARAMS_MAZE_HPP
#define EXAMPLE_PYTORCH_PARAMS_MAZE_HPP

#include <modules/nn2/gen_dnn_ff.hpp>
#include <modules/nn2/phen_dnn.hpp>
#include <modules/nn2/gen_mlp.hpp>


using namespace sferes::gen::evo_float;

struct ParamsMaze {
    SFERES_CONST size_t update_period = 10;
    SFERES_CONST size_t image_width = 50;
    SFERES_CONST size_t image_height = 50;
    SFERES_CONST int resolution = 5000; // influences l; targetted size of pop -> 5000
    SFERES_CONST size_t update_frequency = 10; // -1 means exponentially decaying update frequency
    SFERES_CONST size_t times_downsample = 4; // for taking the image

    SFERES_CONST bool use_colors = true;

    struct nov {
        static double l;
        SFERES_CONST double k = 15;
        SFERES_CONST double eps = 0.1;
    };

    struct pop {
        SFERES_CONST size_t size = 128;
        SFERES_CONST size_t nb_gen = 15001;
        SFERES_CONST size_t dump_period = 500;
    };

    struct evo_float {
        SFERES_CONST float mutation_rate = 0.1f;
        SFERES_CONST float cross_rate = 0.1f;
        SFERES_CONST mutation_t mutation_type = polynomial;
        SFERES_CONST cross_over_t cross_over_type = sbx;
        SFERES_CONST float eta_m = 15.0f;
        SFERES_CONST float eta_c = 15.0f;
    };
    struct parameters {
        // maximum value of parameters
        SFERES_CONST float min = -5.0f;
        // minimum value
        SFERES_CONST float max = 5.0f;
    };
    struct dnn {
        SFERES_CONST size_t nb_inputs = 5;
        SFERES_CONST size_t nb_outputs = 2;
//                SFERES_CONST size_t min_nb_neurons	= 4;
//                SFERES_CONST size_t max_nb_neurons	= 5;
//                SFERES_CONST size_t min_nb_conns	= 50;
//                SFERES_CONST size_t max_nb_conns	= 101;

        SFERES_CONST float m_rate_add_conn = 1.f;
        SFERES_CONST float m_rate_del_conn = 0.05f;
        SFERES_CONST float m_rate_change_conn = 0.1f;
        SFERES_CONST float m_rate_add_neuron = 0.2f;
        SFERES_CONST float m_rate_del_neuron = 0.05f;

        SFERES_CONST int io_param_evolving = true;
        SFERES_CONST sferes::gen::dnn::init_t init = sferes::gen::dnn::init_t::ff;
    };
    struct mlp {
        SFERES_CONST size_t layer_0_size = 4;
        SFERES_CONST size_t layer_1_size = 4;
    };

    struct qd {
        SFERES_CONST size_t behav_dim = 5;
    };

    struct stat {
        SFERES_CONST size_t save_images_period = 500;
        SFERES_CONST size_t period_saving_individual_in_population = 5;
    };
};

double ParamsMaze::nov::l;


#endif //EXAMPLE_PYTORCH_PARAMS_MAZE_HPP
