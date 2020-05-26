//
// Created by Luca Grillotti on 29/04/2020.
//

#ifndef PARAMS_HPP
#define PARAMS_HPP

#include <modules/nn2/gen_dnn_ff.hpp>
#include <modules/nn2/phen_dnn.hpp>
#include <modules/nn2/gen_mlp.hpp>


using namespace sferes::gen::evo_float;

struct Params {
    struct sim {
    SFERES_CONST double ROOM_H = 5;
    SFERES_CONST double ROOM_W = 5;

    // fixed at first or random initialised from main?
    SFERES_CONST double start_x = 3.13;
    SFERES_CONST double start_y = 3.13;
    SFERES_CONST size_t trajectory_length = 50;
    // 2D
    SFERES_CONST int num_trajectory_elements = 2 * trajectory_length;
    };

    struct random {
    SFERES_CONST double pct_random = 0.2;
    SFERES_CONST size_t max_num_random = 1;
    SFERES_CONST bool is_random_dpf = false;
    };

    struct ae {
    SFERES_CONST size_t batch_size = 256;
    SFERES_CONST size_t nb_epochs = 10000;
    SFERES_CONST float convergence_epsilon = 0.0000001;
    SFERES_CONST float CV_fraction = 0.75;
    SFERES_CONST float learning_rate = 1e-4;

    // network neurons        
    // input = qd::gen_dim
    SFERES_CONST size_t en_hid_dim1 = 10;
    // latent_dim = qd::behav_dim
    SFERES_CONST size_t de_hid_dim1 = 10;
    SFERES_CONST size_t de_hid_dim2 = 30;
    // output_dim = sim::trajectory_length
    };
    
    SFERES_CONST size_t discretisation = 20;
    
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
        // this gets used in parameters.hpp to transform the genotype to phenotype
        // maximum value of parameters
        // 0 not needed since minimum for genotype is 0 by default
        SFERES_CONST double max_dpf = 0.5f;
        // minimum value
        SFERES_CONST double max_angle = 2 * M_PI;

        // remove these 2 once phen is fully setup
        SFERES_CONST double min = 0.5f;
        SFERES_CONST double max = 0.5f;
    };

    struct qd {
        SFERES_CONST size_t gen_dim = 2;
        SFERES_CONST size_t behav_dim = 5;
    };

    struct stat {
        SFERES_CONST size_t save_images_period = 500;
        SFERES_CONST size_t period_saving_individual_in_population = 5;
    };
};

double Params::nov::l;


#endif //PARAMS_HPP
