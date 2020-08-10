//
// Created by Andy Wang
//

#ifndef SFERES2_STAT_DISTANCES_HPP
#define SFERES2_STAT_DISTANCES_HPP

#include <sferes/stat/stat.hpp>
#include <numeric>

namespace sferes {
    namespace stat {

        SFERES_STAT(Distances, Stat)
        {
        public:
            Distances(){}

            template<typename EA>
            void refresh(EA &ea) 
            {
                if ((ea.gen() % Params::stat::save_diversity == 0)) 
                {
                   std::string prefix = "distances" + boost::lexical_cast<std::string>(ea.gen());
                    _write_distances(prefix, ea);
                }
            }

            template<typename EA>
            void _write_distances(const std::string &prefix, const EA &ea) 
            {
                std::string fname = ea.res_dir() + "/" + prefix + std::string(".dat");
                std::cout << "writing... " << fname << std::endl;

                Eigen::VectorXf distances(ea.pop().size());
                Eigen::VectorXf distances_excl_zero(ea.pop().size());

                std::array<std::vector<float>, Params::nov::discretisation * Params::nov::discretisation> distances_per_bucket;

                std::vector<int> indices_moved;
                std::array<std::vector<int>, Params::nov::discretisation * Params::nov::discretisation> indices_per_bucket;

                size_t moved_counter{0};
                for (int i{0}; i < ea.pop().size(); ++i)
                {
                    float distance;
                    bool moved{false};
                    int bucket_index = ea.pop()[i]->fit().calculate_distance(distance, moved);
                    distances[i] = distance;
                    indices_per_bucket[bucket_index].push_back(i);
                    if (moved)
                    {
                        indices_moved.push_back(i);
                        distances_per_bucket[bucket_index].push_back(distance);
                        distances_excl_zero[moved_counter] = distance;
                        ++moved_counter;
                    }
                }
                
                Eigen::VectorXf var_grid(Params::nov::discretisation * Params::nov::discretisation);
                var_grid.fill(-20);

                for (int i{0}; i < Params::nov::discretisation * Params::nov::discretisation; ++i)
                {
                    if (distances_per_bucket[i].size() == 0)
                        {continue;}
                    else if (distances_per_bucket[i].size() == 1)
                    {
                        var_grid[i] = 0;
                        continue;
                    }

                    var_grid[i] = 0;
                    float mean_dist = std::accumulate(distances_per_bucket[i].begin(), distances_per_bucket[i].end(), 0.f) / distances_per_bucket[i].size();
                    
                    for (float &dist : distances_per_bucket[i])
                        {var_grid[i] += std::pow(dist - mean_dist, 2);}
                        
                    var_grid[i] /= (distances_per_bucket[i].size() - 1);
                }

                Eigen::VectorXf min_max_grid(Params::nov::discretisation * Params::nov::discretisation);
                min_max_grid.fill(-20);

                for (int i{0}; i < Params::nov::discretisation * Params::nov::discretisation; ++i)
                {
                    if (distances_per_bucket[i].size() == 0)
                        {continue;}
                    else if (distances_per_bucket[i].size() == 1)
                    {
                        min_max_grid[i] = 0;
                        continue;
                    }

                     auto minmax = std::minmax_element(distances_per_bucket[i].begin(), distances_per_bucket[i].end());
                     min_max_grid[i] = *(minmax.second) - *(minmax.first);
                }

                float mean_distance = distances.mean();
                float var_distance = (distances.array() - mean_distance).square().sum() / (ea.pop().size() - 1);

                float mean_distance_moved = distances_excl_zero.segment(0, moved_counter).mean();
                float var_distance_moved = (distances_excl_zero.segment(0, moved_counter).array() - mean_distance_moved).square().sum() / (moved_counter - 1);

                std::ofstream ofs(fname.c_str());
                ofs.precision(17);
                Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", ", "", "");
                ofs << "Mean Distance, Var Distance, Mean Distance NonZero, Var Distance NonZero, Moved/Total\nVar Grid, MinMax Grid, Indices of moved, Indices per bucket, Distances per solution\n";
                ofs << mean_distance << ", " << var_distance << ", " << mean_distance_moved << ", " << var_distance_moved << ", " << moved_counter << "/" << ea.pop().size() << "\n";
                ofs << var_grid.format(CommaInitFmt) << "\n";
                ofs << min_max_grid.format(CommaInitFmt) << "\n";

                for (int &i : indices_moved)
                    {ofs << i << " ";}

                for (int i{0}; i < Params::nov::discretisation * Params::nov::discretisation; ++i)
                {
                    ofs << "\n" << i;
                    for (int index : indices_per_bucket[i])
                        {ofs << ", " << index;}
                }
                ofs << "\n";
                for (int i{0}; i < distances.size(); ++i)
                    {ofs << distances[i] << ", ";}
            }
        };
    }
}


#endif
