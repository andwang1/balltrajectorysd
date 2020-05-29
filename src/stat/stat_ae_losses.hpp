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
                std::cout << "writing..." << fname << std::endl;

                // retrieve all phenotypes and trajectories                
                matrix_t phen, traj;
                std::vector<int> is_traj;
                boost::fusion::at_c<0>(ea.fit_modifier()).get_phen(ea.pop(), phen);
                boost::fusion::at_c<0>(ea.fit_modifier()).get_trajectories(ea.pop(), traj, is_traj);
                
                Eigen::VectorXi is_trajectory;
                boost::fusion::at_c<0>(ea.fit_modifier()).get_network_loader()->vector_to_eigen(is_traj, is_trajectory);

                matrix_t descriptors, recon_loss, recon_loss_unred, reconstruction, L2_loss, KL_loss, decoder_var;
                boost::fusion::at_c<0>(ea.fit_modifier()).get_network_loader()->get_stats(phen, traj, is_trajectory, descriptors, recon_loss, recon_loss_unred, reconstruction, L2_loss, KL_loss, decoder_var);

                

                std::ofstream ofs(fname.c_str(), std::ofstream::app);
                ofs.precision(17);
                double recon = recon_loss.mean();

                #ifdef VAE
                double L2 = L2_loss.mean();
                double KL = KL_loss.mean();
                double var = decoder_var.mean();
                ofs << ea.gen() << ", " << recon << ", " << L2 << ", " << KL << ", " << var;
                #else
                ofs << ea.gen() << ", " << recon;
                #endif

                // training frequency
                if (Params::update::update_frequency == -1) 
                {
                    if (Params::update::update_period > 0 && 
                       (ea.gen() == 1 || ea.gen() == last_update + Params::update::update_period * std::pow(2, update_id - 1))) 
                    {
                        ofs << ", " << boost::fusion::at_c<0>(ea.fit_modifier()).get_network_loader()->get_epochs_trained() << "/" << Params::ae::nb_epochs << ", IS_TRAIN";
                        update_id++;
                    }
                } 
                else if (ea.gen() > 0) 
                {
                    if ((ea.gen() % Params::update::update_frequency == 0) || ea.gen() == 1) 
                    ofs << ", " << boost::fusion::at_c<0>(ea.fit_modifier()).get_network_loader()->get_epochs_trained() << "/" << Params::ae::nb_epochs << ", IS_TRAIN";
                }
                ofs << "\n";
            }
        private:
        size_t last_update{0};
        size_t update_id{0};
        };

    }
}


#endif
