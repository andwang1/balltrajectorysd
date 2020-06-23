//
// Created by Andy Wang
//

#ifndef SFERES2_STAT_AE_LOSSES_HPP
#define SFERES2_STAT_AE_LOSSES_HPP

#include <sferes/stat/stat.hpp>

namespace sferes {
    namespace stat {

        SFERES_STAT(Losses, Stat)
        {
        public:
            typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> matrix_t;

            template<typename EA>
            void refresh(EA &ea) 
            {
                std::string prefix = "ae_loss";
                _write_losses(prefix, ea);
            }
            

            template<typename EA>
            void _write_losses(const std::string &prefix, const EA &ea) {

                std::string fname = ea.res_dir() + "/" + prefix + std::string(".dat");
                std::cout << "writing... " << fname << std::endl;

                // retrieve all phenotypes and trajectories                
                matrix_t gen, traj;
                std::vector<int> is_traj;
                boost::fusion::at_c<0>(ea.fit_modifier()).get_geno(ea.pop(), gen);
                boost::fusion::at_c<0>(ea.fit_modifier()).get_trajectories(ea.pop(), traj, is_traj);
                
                Eigen::VectorXi is_trajectory;
                boost::fusion::at_c<0>(ea.fit_modifier()).get_network_loader()->vector_to_eigen(is_traj, is_trajectory);

                matrix_t descriptors, recon_loss, recon_loss_unred, reconstruction, L2_loss, L2_loss_real_trajectories, KL_loss, decoder_var;
                boost::fusion::at_c<0>(ea.fit_modifier()).get_stats(gen, traj, is_trajectory, descriptors, reconstruction, recon_loss, recon_loss_unred, 
                                                                    L2_loss, L2_loss_real_trajectories, KL_loss, decoder_var);

                std::ofstream ofs(fname.c_str(), std::ofstream::app);
                ofs.precision(17);
                double recon = recon_loss.mean();
                double L2_real_traj = L2_loss_real_trajectories.mean();
                
                #ifdef VAE
                // these three are unreduced, need row wise sum and then mean
                double L2 = L2_loss.rowwise().sum().mean();
                double KL = KL_loss.rowwise().sum().mean();
                double var = decoder_var.rowwise().sum().mean();

                // retrieve trajectories without any interference from random observations
                matrix_t undisturbed_traj(ea.pop().size(), Params::sim::num_trajectory_elements);
                for (size_t i{0}; i < ea.pop().size(); ++i)
                {undisturbed_traj.row(i) = ea.pop()[i]->fit().get_undisturbed_trajectory();}
                double L2_undisturbed_real_traj = (undisturbed_traj - reconstruction).array().square().rowwise().sum().mean();

                ofs << ea.gen() << ", " << recon << ", " << L2 << ", " << KL << ", " << var << ", " << L2_real_traj << ", " << L2_undisturbed_real_traj;
                #else

                #ifdef AURORA
                ofs << ea.gen() << ", " << recon << ", " << L2_real_traj;
                #else // AE

                matrix_t undisturbed_traj(ea.pop().size(), Params::sim::num_trajectory_elements);
                for (size_t i{0}; i < ea.pop().size(); ++i)
                {undisturbed_traj.row(i) = ea.pop()[i]->fit().get_undisturbed_trajectory();}
                double L2_undisturbed_real_traj = (undisturbed_traj - reconstruction).array().square().rowwise().sum().mean();

                ofs << ea.gen() << ", " << recon << ", " << L2_real_traj << ", " << L2_undisturbed_real_traj;
                #endif 

                #endif

                if (boost::fusion::at_c<0>(ea.fit_modifier()).is_train_gen())
                {
                    #ifndef AURORA
                    ofs << ", " << boost::fusion::at_c<0>(ea.fit_modifier()).get_random_extension_ratio() << ", " << boost::fusion::at_c<0>(ea.fit_modifier()).get_network_loader()->get_epochs_trained() << "/" << Params::ae::nb_epochs << ", IS_TRAIN";
                    #else
                    ofs << ", " << boost::fusion::at_c<0>(ea.fit_modifier()).get_network_loader()->get_epochs_trained() << "/" << Params::ae::nb_epochs << ", IS_TRAIN";
                    #endif
                }
                ofs << "\n";
            }
        };

    }
}


#endif
