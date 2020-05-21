//
// Created by Luca Grillotti on 13/01/2020.
//

#ifndef EXAMPLE_PYTORCH_STAT_MODEL_AUTOENCODER_HPP
#define EXAMPLE_PYTORCH_STAT_MODEL_AUTOENCODER_HPP

#include <sferes/stat/stat.hpp>

namespace sferes {
    namespace stat {

        SFERES_STAT(ModelAutoencoder, Stat)
        {
        public:
            template<typename EA>
            void refresh(EA &ea) {
                if (ea.gen() % Params::stat::save_images_period == 0) {
                    std::string name_file = ea.res_dir() + "/"
                                            + "model_autoencoder_gen_"
                                            + boost::lexical_cast<std::string>(ea.gen())
                                            + ".pt";
                    std::cout << "writing... " << name_file << std::endl;
                    torch::save(boost::fusion::at_c<0>(ea.fit_modifier()).get_network_loader()->auto_encoder().ptr(), name_file);
                }
            }
        };

    }
}

#endif //EXAMPLE_PYTORCH_STAT_MODEL_AUTOENCODER_HPP
