//
// Created by Andy Wang
//

#ifndef SFERES2_STAT_DIVERSITY_HPP
#define SFERES2_STAT_DIVERSITY_HPP

#include <sferes/stat/stat.hpp>

namespace sferes {
    namespace stat {

        SFERES_STAT(Diversity, Stat)
        {
        public:
            Diversity():_nums_covered_buckets(Params::nov::discretisation * Params::nov::discretisation){}

            typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> matrix_t;
            typedef std::bitset<Params::nov::discretisation * Params::nov::discretisation> div_t;

            void reset_array()
            {
                for (int i{0}; i < _array_div.size(); ++i)
                    {_array_div[i].reset();}
            }

            template<typename EA>
            void refresh(EA &ea) 
            {
                if ((ea.gen() % Params::stat::save_diversity == 0)) 
                {
                   std::string prefix = "diversity" + boost::lexical_cast<std::string>(ea.gen());
                    _write_diversity(prefix, ea);
                }
            }

            template<typename EA>
            void _write_diversity(const std::string &prefix, const EA &ea) 
            {
                std::string fname = ea.res_dir() + "/" + prefix + std::string(".dat");
                std::cout << "writing..." << fname << std::endl;

                // reset the bitmaps
                reset_array();

                // loop through all objects and get their bitmap
                for (auto it = ea.pop().begin(); it != ea.pop().end(); ++it)
                {
                    div_t indiv_diversity{0};
                    int bucket_index = (*it)->fit().calculate_diversity_bins(indiv_diversity);
                    _array_div[bucket_index] |= indiv_diversity;
                }

                for (int j{0}; j < _nums_covered_buckets.size(); ++j)
                {
                    int buckets_count = _array_div[j].count();
                    _nums_covered_buckets[j] = buckets_count;
                }
                _nums_covered_buckets /= double(Params::nov::discretisation * Params::nov::discretisation);
                double total_diversity = _nums_covered_buckets.sum();

                std::ofstream ofs(fname.c_str());
                ofs.precision(17);
                Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", ", "", "");
                ofs << "First Line is the overall diversity map, Second Line is the total diversity score, Max diversity: "<< Params::nov::discretisation * Params::nov::discretisation <<"\n";
                ofs << total_diversity << "\n";
                ofs << _nums_covered_buckets.format(CommaInitFmt) << std::endl;
            }

        private:
        std::array<div_t, Params::nov::discretisation * Params::nov::discretisation> _array_div;
        Eigen::VectorXd _nums_covered_buckets;
        };

    }
}


#endif
