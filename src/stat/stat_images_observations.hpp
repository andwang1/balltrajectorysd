//
// Created by Luca Grillotti on 30/10/2019.
//

#ifndef SFERES2_STAT_IMAGES_OBSERVATIONS_HPP
#define SFERES2_STAT_IMAGES_OBSERVATIONS_HPP

#include <sferes/stat/stat.hpp>

#include "stat/utils.hpp"

namespace sferes {
    namespace stat {

        SFERES_STAT(ImagesObservations, Stat)
        {
        public:
            void save_specific_index(const boost::shared_ptr<Phen>& indiv, const std::string &prefix_image_indiv) const {
                auto rgb_image = indiv->fit().get_rgb_image();

                if (Params::use_colors) {
                    save_png_image(prefix_image_indiv + "_color.png", rgb_image);
                } else {
                    save_png_image(prefix_image_indiv + "_grayscale.png",
                                                    convert_rgb_to_grayscale(rgb_image));
                }
            }

            template<typename EA>
            void save_images_specific_observations(const std::string &prefix, const EA &ea,
                                                   const std::vector<size_t>& indexes_to_save) const {
                std::cout << "writing..." << prefix << std::endl;
                std::string prefix_image_indiv;

                for (size_t index_to_save : indexes_to_save) {
                    prefix_image_indiv = prefix
                                         + "_indiv_"
                                         + add_leading_zeros(index_to_save);

                    save_specific_index(ea.pop()[index_to_save], prefix_image_indiv);
                }
            }

            template<typename EA>
            void _save_images_observations(const std::string &prefix, const EA &ea) const {
                std::cout << "writing..." << prefix << std::endl;
                std::string prefix_image_indiv;

                size_t index_indiv{0};
                for (auto it = ea.pop().begin(); it != ea.pop().end(); ++it) {
                    if (index_indiv % Params::stat::period_saving_individual_in_population == 0) {
                        prefix_image_indiv = prefix
                                             + "_indiv_"
                                             + add_leading_zeros(index_indiv);

                        save_specific_index(*it, prefix_image_indiv);
                    }
                    ++index_indiv;
                }
            }

            template<typename EA>
            void refresh(EA &ea) {
                if ((ea.gen() % Params::stat::save_images_period == 0) || (ea.gen() == 1)) {
                    std::string prefix = ea.res_dir() + "/"
                                         + "observation_gen_"
                                         + add_leading_zeros(ea.gen());
                    _save_images_observations(prefix, ea);
                }
            }
        };

    }
}


#endif
