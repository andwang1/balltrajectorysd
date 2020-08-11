//
// Created by Andy Wang
//

#ifndef SFERES2_STAT_TRAJ_HPP
#define SFERES2_STAT_TRAJ_HPP

#include <sferes/stat/stat.hpp>

namespace sferes {
    namespace stat {

        SFERES_STAT(Trajectories, Stat)
        {
        public:
            typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> matrix_t;

            template<typename EA>
            void refresh(EA &ea) 
            {
                if ((ea.gen() % Params::stat::save_trajectories == 0) && (ea.gen() > 0)) 
                {
                   std::string prefix = "traj_" + boost::lexical_cast<std::string>(ea.gen());
                    _write_trajectories(prefix, ea);
                }
            }

            template<typename EA>
            void _write_trajectories(const std::string &prefix, const EA &ea) const {

                std::string fname = ea.res_dir() + "/" + prefix + std::string(".dat");
                std::cout << "writing... " << fname << std::endl;

                // retrieve all phenotypes and trajectories                
                matrix_t gen, traj;
                std::vector<int> is_traj;
                boost::fusion::at_c<0>(ea.fit_modifier()).get_geno(ea.pop(), gen);
                boost::fusion::at_c<0>(ea.fit_modifier()).get_trajectories(ea.pop(), traj, is_traj);
                
                // filter out the realised trajectories
                matrix_t filtered_traj;
                std::vector<bool> boundaries;
                Eigen::VectorXi is_trajectory;
                boost::fusion::at_c<0>(ea.fit_modifier()).get_network_loader()->vector_to_eigen(is_traj, is_trajectory);
                boost::fusion::at_c<0>(ea.fit_modifier()).get_network_loader()->filter_trajectories(traj, is_trajectory, filtered_traj, boundaries);
                
                // get all data
                matrix_t descriptors, recon_loss, recon_loss_unred, reconstruction, L2_loss, L2_loss_real_trajectories, KL_loss, encoder_var, decoder_var;
                boost::fusion::at_c<0>(ea.fit_modifier()).get_stats(gen, traj, is_trajectory, descriptors, reconstruction, recon_loss, recon_loss_unred, L2_loss, L2_loss_real_trajectories, KL_loss, encoder_var, decoder_var);
                std::ofstream ofs(fname.c_str());
                ofs.precision(17);
                Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", ", "", "");

                #ifdef AURORA
                ofs << "FORMAT: INDIV_INDEX, DATA\n";
                for (int i{0}; i < reconstruction.rows(); ++i)
                {
                    ofs << i << ", RECON, "  <<  reconstruction.row(i).format(CommaInitFmt) << "\n";
                    ofs << i << ", ACTUAL, " <<  traj.row(i).format(CommaInitFmt) << "\n";
                }

                #else //VAE or AE

                // hack to make the do while loop below work
                boundaries.push_back(true);

                // there are more trajectories than reconstructions as there is only one recon per phen
                size_t traj_index = 0;
                ofs << "FORMAT: INDIV_INDEX, TYPE, DATA\n";
                for (int i{0}; i < reconstruction.rows(); ++i)
                {
                    ofs << i << ", RECON," <<  reconstruction.row(i).format(CommaInitFmt) << "\n";
                    ofs << i << ", RECON_LOSS," <<  recon_loss_unred.row(i).format(CommaInitFmt) << "\n";
                    #ifdef VAE
                    ofs << i << ", KL_LOSS," <<  KL_loss.row(i).format(CommaInitFmt) << "\n";
                    ofs << i << ", DECODER_VAR," <<  decoder_var.row(i).format(CommaInitFmt) << "\n";
                    #endif
                    do
                    {
                        ofs << i << ", ACTUAL," <<  filtered_traj.row(traj_index).format(CommaInitFmt) << "\n";
                        #ifdef VAE
                        ofs << i << ", L2_loss," <<  L2_loss.row(traj_index).format(CommaInitFmt) << "\n";
                        #endif
                        ++traj_index;
                    }
                    while (!boundaries[traj_index]);
                }
                #endif
            }
        };
    }
}

#endif
