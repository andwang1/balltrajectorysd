//
// Created by Luca Grillotti on 29/04/2020.
//

#ifndef PARAMS_HPP
#define PARAMS_HPP

using namespace sferes::gen::evo_float;

struct Params {
    struct sim {
    SFERES_CONST double ROOM_H = 5;
    SFERES_CONST double ROOM_W = 5;

    // fixed at first or random initialised from main?
    SFERES_CONST double start_x = 3.13;
    SFERES_CONST double start_y = 1.17;
    SFERES_CONST double dpf = 0.21;
    SFERES_CONST size_t trajectory_length = 50;
    // 2D
    SFERES_CONST int num_trajectory_elements = 2 * trajectory_length;
    };

    struct random {
    static double pct_random;
    SFERES_CONST size_t max_num_random = 1;
    SFERES_CONST bool is_random_dpf = false;
    };

    struct ae {
    SFERES_CONST size_t batch_size = 64;
    SFERES_CONST size_t nb_epochs = 10000;
    SFERES_CONST size_t min_num_epochs = 0;
    SFERES_CONST size_t running_mean_num_epochs = 5;
    SFERES_CONST float CV_fraction = 0.80;
    SFERES_CONST float learning_rate = 1e-3;
    static bool full_loss;

    // network neurons        
    // input = qd::gen_dim
    SFERES_CONST size_t en_hid_dim1 = 10;
    // latent_dim = qd::behav_dim
    SFERES_CONST size_t de_hid_dim1 = 40;
    SFERES_CONST size_t de_hid_dim2 = 60;
    // output_dim = sim::trajectory_length

    // KL weight
    SFERES_CONST size_t beta = 1;

    // 
    };

    struct update {
    // used in deciding how often to apply dim reduction (and training)
    SFERES_CONST size_t update_frequency = 20; // -1 means exponentially decaying update frequency, how often update BD etc
    SFERES_CONST size_t update_period = 20;
    };

    // influences l; targetted size of pop
    SFERES_CONST int resolution = 10000; 
    
    struct nov {
        static double l;
        SFERES_CONST double k = 15;
        SFERES_CONST double eps = 0.1;
        // the discretisation used to create the diversity bin data
        SFERES_CONST size_t discretisation = 20;
    };

    struct pop {
        SFERES_CONST size_t size = 256;
        SFERES_CONST size_t nb_gen = 20001;
        SFERES_CONST size_t dump_period = 1000;
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
    };

    struct qd {
        SFERES_CONST size_t gen_dim = 2;
        SFERES_CONST size_t behav_dim = 2;
    };

    struct stat {
        SFERES_CONST size_t save_trajectories = 1000;
        SFERES_CONST size_t save_model = 5000;
        SFERES_CONST size_t save_diversity = 500;
    };
};

double Params::nov::l;
double Params::random::pct_random;
bool Params::ae::full_loss;


#endif //PARAMS_HPP
