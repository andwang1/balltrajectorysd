//
// Created by Andy Wang
//

#ifndef SFERES2_STAT_TRAJ_SIMIL_HPP
#define SFERES2_STAT_TRAJ_SIMIL_HPP

#include <sferes/stat/stat.hpp>
#include <numeric>

namespace sferes {
    namespace stat {

        SFERES_STAT(Similarity, Stat)
        {
        public:
            typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> matrix_t;
            Similarity(){}

            template<typename EA>
            void refresh(EA &ea) 
            {
                if ((ea.gen() % Params::stat::save_diversity == 0)) 
                {
                   std::string prefix = "similarities" + boost::lexical_cast<std::string>(ea.gen());
                    _write_similarities(prefix, ea);
                }
            }

            template<typename EA>
            void _write_similarities(const std::string &prefix, const EA &ea) 
            {
                std::string fname = ea.res_dir() + "/" + prefix + std::string(".dat");
                std::cout << "writing... " << fname << std::endl;

                matrix_t observations(ea.pop().size(), Params::sim::num_trajectory_elements);
                matrix_t observations_excl_zero(ea.pop().size(), Params::sim::num_trajectory_elements);
                std::vector<int> indices_moved;
                std::array<std::vector<int>, Params::nov::discretisation * Params::nov::discretisation> indices_per_bucket;

                size_t moved_counter{0};
                for (int i{0}; i < ea.pop().size(); ++i)
                {
                    observations.row(i) = ea.pop()[i]->fit().get_undisturbed_trajectory();
                    if (ea.pop()[i]->fit().moved())
                    {
                        indices_per_bucket[ea.pop()[i]->fit().get_bucket_index(Params::nov::discrete_length_x, Params::nov::discrete_length_y, Params::nov::discretisation)].push_back(i);
                        indices_moved.push_back(i);
                        
                        observations_excl_zero.row(moved_counter) = observations.row(i);
                        ++moved_counter;
                    }
                }

                Eigen::VectorXf variance = (observations.rowwise() - observations.colwise().mean()).array().square().colwise().sum() / (ea.pop().size() - 1);
                Eigen::VectorXf variance_excl_zero = (observations_excl_zero.block(0, 0, moved_counter, Params::sim::num_trajectory_elements).rowwise() - observations_excl_zero.block(0, 0, moved_counter, Params::sim::num_trajectory_elements).colwise().mean()).array().square().colwise().sum().array() / (moved_counter - 1);

                Eigen::VectorXf var_grid(Params::nov::discretisation * Params::nov::discretisation);
                var_grid.fill(-20);

                Eigen::VectorXi freq_grid(Params::nov::discretisation * Params::nov::discretisation);
                freq_grid.fill(0);

                for (int i{0}; i < Params::nov::discretisation * Params::nov::discretisation; ++i)
                {
                    if (indices_per_bucket[i].size() == 0)
                        {continue;}
                    else if (indices_per_bucket[i].size() == 1)
                    {
                        var_grid[i] = 0;
                        freq_grid[i] = 1;
                        continue;
                    }
                    
                    matrix_t bucket_observations(indices_per_bucket[i].size(), Params::sim::num_trajectory_elements);
                    int row_counter{0};
                    for (int &index : indices_per_bucket[i])
                    {
                        bucket_observations.row(row_counter) = observations.row(index);
                        ++row_counter;
                    }

                    Eigen::VectorXf bucket_coord_var = (bucket_observations.rowwise() - bucket_observations.colwise().mean()).array().square().colwise().sum() / (indices_per_bucket[i].size() - 1);
                    var_grid[i] = 0;
                    for (int j{0}; j < Params::sim::num_trajectory_elements; j += 2)
                        {var_grid[i] += bucket_coord_var.segment<2>(j).mean();}

                    var_grid[i] /= Params::sim::trajectory_length;
                    freq_grid[i] = indices_per_bucket[i].size();
                }
                
                std::ofstream ofs(fname.c_str());
                ofs.precision(17);
                Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", ", "", "");
                ofs << "Mean Coordwise Var, Mean Coordwise Var excl. Zero, Moved/Total\nPointwise Var grid, Freq. grid\n";
                ofs << variance.mean() << ", " << variance_excl_zero.mean() << ", " << moved_counter << "/" << ea.pop().size() << "\n";
                ofs << var_grid.format(CommaInitFmt) << "\n";
                ofs << freq_grid.format(CommaInitFmt);
            }
        };
    }
}


#endif
