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
                if ((ea.gen() % Params::stat::save_trajectories == 0) || (ea.gen() == 1) ) 
                {
                   std::string prefix = "traj_" + boost::lexical_cast<std::string>(ea.gen());
                    _write_trajectories(prefix, ea);
                }
                
                std::string prefix = "ae_loss";
                _write_losses(prefix, ea);
            }

            template<typename EA>
            void _write_trajectories(const std::string &prefix, const EA &ea) const {

                std::string fname = ea.res_dir() + "/" + prefix + std::string(".dat");
                std::cout << "writing..." << fname << std::endl;

                // retrieve all phenotypes and trajectories                
                matrix_t phen, traj;
                std::vector<int> is_traj;
                boost::fusion::at_c<0>(ea.fit_modifier()).get_phen(ea.pop(), phen);
                boost::fusion::at_c<0>(ea.fit_modifier()).get_trajectories(ea.pop(), traj, is_traj);
                
                // filter out the realised trajectories
                matrix_t filtered_traj;
                std::vector<bool> boundaries;
                Eigen::VectorXi is_trajectory;
                boost::fusion::at_c<0>(ea.fit_modifier()).get_network_loader()->vector_to_eigen(is_traj, is_trajectory);
                boost::fusion::at_c<0>(ea.fit_modifier()).get_network_loader()->filter_trajectories(traj, is_trajectory, filtered_traj, boundaries);
                
                // get the reconstruction
                matrix_t reconstruction;
                boost::fusion::at_c<0>(ea.fit_modifier()).get_reconstruction(phen, traj, is_traj, reconstruction);

                std::ofstream ofs(fname.c_str());
                ofs.precision(17);
                Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", ", "", "");

                // std::cout << "pop" << ea.pop().size() << std::endl;
                // std::cout << "traj" << filtered_traj.rows() << ",c " << filtered_traj.cols() << std::endl;
                // std::cout << "reconstruction" << reconstruction.rows() << ",c " << reconstruction.cols() << std::endl;

                // hack to make the do while loop below work
                boundaries.push_back(true);
            
                // there are more trajectories than reconstructions as there is only one recon per phen
                size_t traj_index = 0;
                ofs << "FORMAT: INDIV_INDEX, RECON/ACTUAL, DATA\n";
                for (int i{0}; i < reconstruction.rows(); ++i)
                {
                    ofs << i << ", RECON," <<  reconstruction.row(i).format(CommaInitFmt) << std::endl;
                    do
                    {
                        ofs << i << ", ACTUAL," <<  filtered_traj.row(traj_index).format(CommaInitFmt) << std::endl;
                        ++traj_index;
                    }
                    while (!boundaries[traj_index]);
                }
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

                float recon_loss = boost::fusion::at_c<0>(ea.fit_modifier()).get_network_loader()->get_avg_recon_loss(phen, traj, is_trajectory);

                std::ofstream ofs(fname.c_str(), std::ofstream::app);
                ofs.precision(17);
                ofs << recon_loss << ", ";

                // print a second line with only losses after training

                // training frequency
                if (Params::update::update_frequency == -1) 
                {
                    if (Params::update::update_period > 0 && 
                       (ea.gen() == 1 || ea.gen() == last_update + Params::update::update_period * std::pow(2, update_id - 1))) 
                    {
                        std::string train_fname = ea.res_dir() + "/" + prefix + "_train" + std::string(".dat");
                        std::ofstream ofs_train(train_fname.c_str(), std::ofstream::app);
                        ofs_train.precision(17);
                        ofs_train << recon_loss << ", ";
                        update_id++;
                    }
                } 
                else if (ea.gen() > 0) 
                {
                    if ((ea.gen() % Params::update::update_frequency == 0) || ea.gen() == 1) 
                    {
                        std::string train_fname = ea.res_dir() + "/" + prefix + "_train" + std::string(".dat");
                        std::ofstream ofs_train(train_fname.c_str(), std::ofstream::app);
                        ofs_train.precision(17);
                        ofs_train << recon_loss << ", ";
                    }
                }
            }
        private:
        size_t last_update{0};
        size_t update_id{0};
        };

    }
}


#endif
