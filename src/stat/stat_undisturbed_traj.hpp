//
// Created by Andy Wang
//

#ifndef SFERES2_STAT_UNDIST_TRAJ_HPP
#define SFERES2_STAT_UNDIST_TRAJ_HPP

#include <sferes/stat/stat.hpp>

namespace sferes {
    namespace stat {

        SFERES_STAT(UndisturbedTrajectories, Stat)
        {
        public:
            typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> matrix_t;

            template<typename EA>
            void refresh(EA &ea) 
            {
                if ((ea.gen() % Params::stat::save_trajectories == 0) && (ea.gen() > 0)) 
                {
                   std::string prefix = "undist_traj_" + boost::lexical_cast<std::string>(ea.gen());
                    _write_trajectories(prefix, ea);
                }
            }

            template<typename EA>
            void _write_trajectories(const std::string &prefix, const EA &ea) const {

                std::string fname = ea.res_dir() + "/" + prefix + std::string(".dat");
                std::cout << "writing... " << fname << std::endl;

                std::ofstream ofs(fname.c_str());
                ofs.precision(17);
                Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", ", "", "");

                for (int i{0}; i < ea.pop().size(); ++i)
                    {ofs << ea.pop()[i]->fit().get_undisturbed_trajectory().format(CommaInitFmt) << "\n";}
            }
        };
    }
}

#endif
