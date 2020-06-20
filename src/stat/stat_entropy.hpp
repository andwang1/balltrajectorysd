//
// Created by Andy Wang
//

#ifndef SFERES2_STAT_ENTROPY_HPP
#define SFERES2_STAT_ENTROPY_HPP

#include <sferes/stat/stat.hpp>
#include <numeric>

namespace sferes {
    namespace stat {

        SFERES_STAT(Entropy, Stat)
        {
        public:
            typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> matrix_t;
            Entropy(){}

            template<typename EA>
            void refresh(EA &ea) 
            {
                if ((ea.gen() % Params::stat::save_diversity == 0)) 
                {
                   std::string prefix = "entropy" + boost::lexical_cast<std::string>(ea.gen());
                    _write_entropy(prefix, ea);
                }
            }

            template<typename EA>
            void _write_entropy(const std::string &prefix, const EA &ea) 
            {
                std::string fname = ea.res_dir() + "/" + prefix + std::string(".dat");
                std::cout << "writing... " << fname << std::endl;

                size_t pop_size = ea.pop().size();

                matrix_t observations(pop_size, Params::sim::num_trajectory_elements);
                matrix_t observations_excl_zero(pop_size, Params::sim::num_trajectory_elements);
                std::array<std::vector<int>, Params::stat::entropy_discretisation * Params::stat::entropy_discretisation> indices_per_bucket;

                size_t moved_counter{0};
                for (int i{0}; i < pop_size; ++i)
                {
                    observations.row(i) = ea.pop()[i]->fit().get_undisturbed_trajectory();
                    if (ea.pop()[i]->fit().moved())
                    {
                        indices_per_bucket[ea.pop()[i]->fit().get_bucket_index(Params::stat::ent_discrete_length_x, Params::stat::ent_discrete_length_y, Params::stat::entropy_discretisation)].push_back(i);
                        
                        observations_excl_zero.row(moved_counter) = observations.row(i);
                        ++moved_counter;
                    }
                }

                assert((Params::stat::ent_discrete_length_x - Params::stat::ent_discrete_length_y < 1e-5) && "Entropy Calc depends on same discretisation");
                observations /= Params::stat::ent_discrete_length_x;
                matrix_t observations_excl_zero_discretised_f = observations_excl_zero.block(0, 0, moved_counter, Params::sim::num_trajectory_elements) / Params::stat::ent_discrete_length_x;

                Eigen::Matrix<int, Eigen::Dynamic, Params::sim::num_trajectory_elements> observations_discretised = observations.cast<int>();
                Eigen::Matrix<int, Eigen::Dynamic, Params::sim::num_trajectory_elements> observations_excl_zero_discretised = observations_excl_zero_discretised_f.cast<int>();

                Eigen::VectorXf entropy_values(Params::sim::trajectory_length);
                Eigen::VectorXf entropy_excl_zero(Params::sim::trajectory_length);
                entropy_values.fill(0);
                entropy_excl_zero.fill(0);

                // get the bucket for each point in the trajectories
                for (int i{0}; i < Params::sim::num_trajectory_elements; i += 2)
                {
                    Eigen::VectorXi observations_buckets = observations_discretised.col(i) + observations_discretised.col(i + 1) * Params::stat::entropy_discretisation;
                    Eigen::VectorXi observations_buckets_excl_zero = observations_excl_zero_discretised.col(i) + observations_excl_zero_discretised.col(i + 1) * Params::stat::entropy_discretisation;

                    // unique values
                    std::unordered_set<int> unique_buckets(observations_buckets.data(), observations_buckets.data() + observations_buckets.size());
                    std::unordered_set<int> unique_buckets_excl_zero(observations_buckets_excl_zero.data(), observations_buckets_excl_zero.data() + observations_buckets_excl_zero.size());
                    
                    // entropy calc
                    for (int j : unique_buckets)
                    {
                        double p = std::count(observations_buckets.data(), observations_buckets.data() + observations_buckets.size(), j) / double (pop_size);
                        entropy_values[i / 2] -= p * log2(p);
                    }

                    for (int j : unique_buckets_excl_zero)
                    {
                        double p = std::count(observations_buckets_excl_zero.data(), observations_buckets_excl_zero.data() + observations_buckets_excl_zero.size(), j) / double (moved_counter);
                        entropy_excl_zero[i / 2] -= p * log2(p);
                    }
                }

                // loop through the buckets individually and populate the grid
                Eigen::VectorXf avg_entropy_grid(Params::stat::entropy_discretisation * Params::stat::entropy_discretisation);
                avg_entropy_grid.fill(-20);

                Eigen::VectorXi freq_grid(Params::stat::entropy_discretisation * Params::stat::entropy_discretisation);
                freq_grid.fill(0);

                for (int i{0}; i < Params::stat::entropy_discretisation * Params::stat::entropy_discretisation; ++i)
                {
                    if (indices_per_bucket[i].size() == 0)
                        {continue;}
                    else if (indices_per_bucket[i].size() == 1)
                    {
                        avg_entropy_grid[i] = 0;
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

                    matrix_t bucket_observations_discretised_f = bucket_observations / Params::stat::ent_discrete_length_x;
                    Eigen::Matrix<int, Eigen::Dynamic, Params::sim::num_trajectory_elements> bucket_observations_discretised = bucket_observations_discretised_f.cast<int>();
                    Eigen::VectorXf bucket_entropy_values(Params::sim::trajectory_length);
                    bucket_entropy_values.fill(0);

                    // get the bucket for each point in the trajectories
                    for (int i{0}; i < Params::sim::trajectory_length; i += 2)
                    {
                        Eigen::VectorXi observations_buckets = bucket_observations_discretised.col(i) + bucket_observations_discretised.col(i + 1) * Params::stat::entropy_discretisation;
                        std::unordered_set<int> unique_buckets(observations_buckets.data(), observations_buckets.data() + observations_buckets.size());

                        for (int j : unique_buckets)
                        {
                            double p = std::count(observations_buckets.data(), observations_buckets.data() + observations_buckets.size(), j) / double(observations_buckets.size());
                            bucket_entropy_values[i / 2] -= p * log2(p);
                        }
                    }
                    avg_entropy_grid[i] = bucket_entropy_values.mean();
                    freq_grid[i] = indices_per_bucket[i].size();
                }
                
                std::ofstream ofs(fname.c_str());
                ofs.precision(17);
                Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", ", "", "");
                ofs << "Mean Entropy, Mean Entropy excl. Zero, Moved/Total\nEntropy grid\n";
                ofs << entropy_values.mean() << ", " << entropy_excl_zero.mean() << ", " << moved_counter << "/" << pop_size << "\n";
                ofs << avg_entropy_grid.format(CommaInitFmt) << "\n";
                ofs << freq_grid.format(CommaInitFmt);
            }
        };
    }
}


#endif
