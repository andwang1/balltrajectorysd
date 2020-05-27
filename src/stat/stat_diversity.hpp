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
            Diversity():nums_covered_buckets(Params::nov::discretisation * Params::nov::discretisation){}

            typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> matrix_t;
            typedef std::bitset<Params::nov::discretisation * Params::nov::discretisation> div_t;

            void reset_array()
            {
                for (int i{0}; i < array_div.size(); ++i)
                {array_div[i].reset();}
            }

            template<typename EA>
            void refresh(EA &ea) 
            {
                if ((ea.gen() % Params::stat::save_diversity == 0) || (ea.gen() == 1) ) 
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

                int i{0};
                for (auto it = ea.pop().begin(); it != ea.pop().end(); ++it)
                {
                    div_t indiv_diversity{0};
                    int bucket_index = (*it)->fit().calculate_diversity_bins(indiv_diversity);
                    // std::cout << indiv_diversity << std::endl;
                    // std::cout << "INDEX" << bucket_index << std::endl;
                    array_div[bucket_index] |= indiv_diversity;
                    ++i;
                }

                for (int j{0}; j < nums_covered_buckets.size(); ++j)
                {
                    int buckets_count = array_div[j].count();
                    nums_covered_buckets[j] = buckets_count;
                }
                nums_covered_buckets /= double(Params::nov::discretisation * Params::nov::discretisation);
                double total_diversity = nums_covered_buckets.sum();

                std::ofstream ofs(fname.c_str());
                ofs.precision(17);
                Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", ", "", "");
                ofs << "First Line is the overall diversity map, Second Line is the total diversity score, Max diversity: "<< Params::nov::discretisation * Params::nov::discretisation <<"\n";
                ofs << total_diversity << "\n";
                ofs << nums_covered_buckets.format(CommaInitFmt) << std::endl;
            }

            // attempt to do it for the full container
            // template<typename EA>
            // void _write_diversity(const std::string &prefix, EA &ea) 
            // {
            //     std::string fname = ea.res_dir() + "/" + prefix + std::string(".dat");
            //     std::cout << "writing..." << fname << std::endl;

            //     // reset the bitmaps
            //     reset_array();

            //     // loop through all objects and get their bitmap
            //     std::vector<boost::shared_ptr<Phen> > content;

            //     ea.container().get_full_content(content);

            //     for (int i{0}; i < content.size(); ++i)
            //     {
            //         div_t indiv_diversity{0};
            //         int bucket_index = (content[i])->fit().calculate_diversity_bins(indiv_diversity);
            //         array_div[bucket_index] |= indiv_diversity;
            //     }

            //     for (int i{0}; nums_covered_buckets.size(); ++i)
            //     {
            //         nums_covered_buckets[i] = array_div[i].count();
            //     }

            //     nums_covered_buckets /= double(Params::nov::discretisation * Params::nov::discretisation);
                
            //     double total_diversity = nums_covered_buckets.sum();

            //     std::ofstream ofs(fname.c_str());
            //     ofs.precision(17);
            //     Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", ", "", "");
            //     ofs << "First Line is the overall diversity map, Second Line is the total diversity score\n";
            //     ofs << total_diversity << "\n";
            //     ofs << nums_covered_buckets.format(CommaInitFmt) << std::endl;
            // }

            
        private:
        std::array<std::bitset<Params::nov::discretisation * Params::nov::discretisation>, Params::nov::discretisation * Params::nov::discretisation> array_div;
        Eigen::VectorXd nums_covered_buckets;
        };

    }
}


#endif
