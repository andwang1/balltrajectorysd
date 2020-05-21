//
// Created by Luca Grillotti on 24/10/2019.
//

#ifndef SFERES2_DIMENSIONALITY_REDUCTION_HPP
#define SFERES2_DIMENSIONALITY_REDUCTION_HPP

#include <memory>
#include <sferes/stc.hpp>
#include "preprocessor.hpp"

namespace sferes {
    namespace modif {
        template<typename Phen, typename Params, typename NetworkLoader>
        class DimensionalityReduction {
        public:
            typedef Phen phen_t;
            typedef boost::shared_ptr<Phen> indiv_t;
            typedef typename std::vector<indiv_t> pop_t;
            typedef std::vector<std::pair<std::vector<double>, float>> stat_t;

            DimensionalityReduction() : last_update(0), update_id(0) {
                _prep.init();
                network = std::make_unique<NetworkLoader>();
            }

            using Mat = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

            // defining new matrix for better precision when calculating the new minimum distance l
            using Mat_dist = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

            template<typename EA>
            void apply(EA &ea) {
                /*
                 * Basis function used for applying the modifyer to the population
                 * first update descriptors (only every k steps)
                 * assign those values to the population (every step)
                 * */

                if (Params::update_frequency == -1) {           // NOTE: THERE's BEEN A CHANGE to update_period
                    if (Params::update_period > 0 &&
                        (ea.gen() == 1 ||
                         ea.gen() == last_update + Params::update_period * std::pow(2, update_id - 1))) {
                        update_id++;
                        last_update = ea.gen();
                        update_descriptors(ea);
                    }
                } else if (ea.gen() > 0) {
                    if ((ea.gen() % Params::update_frequency == 0) || ea.gen() == 1) {
                        update_descriptors(ea);
                    }
                }

                if (!ea.offspring().size()) return;

                assign_descriptor_to_population(ea, ea.offspring());
            }

            template<typename EA>
            void update_descriptors(EA &ea) {
                Mat data;
                collect_dataset(data, ea, true); // gather the data from the indiv in the archive into a dataset
                train_network(data);
                update_container(ea);  // clear the archive and re-fill it using the new network
                ea.pop_advers.clear(); // clearing adversarial examples
            }

            template<typename EA>
            void collect_dataset(Mat &data,
                    EA &ea,
                    const std::vector<typename EA::indiv_t>& content,
                    bool training = false) const {

                size_t pop_size = content.size();
                Mat pop_data, advers_data;
                advers_data.resize(0, 0);

                get_data(content, pop_data);


                if (training && ea.pop_advers.size()) {
                    get_data(ea.pop_advers, advers_data);
                    int rows = pop_data.rows() + advers_data.rows();
                    int cols = pop_data.cols();
                    data.resize(rows, cols);
                    data << pop_data,
                            advers_data;
                } else {
                    int rows = pop_data.rows();
                    int cols = pop_data.cols();
                    data.resize(rows, cols);
                    data << pop_data;
                }

                if (training) {
                    std::cout << "training set is composed of " << data.rows() << " samples  ("
                              << ea.gen() << " archive size : " << pop_size << ")" << std::endl;
                }
            }

            template<typename EA>
            void collect_dataset(Mat &data, EA &ea, bool training = false) const {
                std::vector<typename EA::indiv_t> content;
                if (ea.gen() > 0) {
                    ea.container().get_full_content(content);
                } else {
                    content = ea.offspring();
                }
                collect_dataset(data, ea, content, training);
            }

            void train_network(const Mat &data) {
                // we change the data normalisation each time we train/refine network, could cause small changes in loss between two trainings.
                _prep.init(data);
                Mat scaled_data;
                _prep.apply(data, scaled_data);
                float final_entropy = network->training(scaled_data);
            }

            template<typename EA>
            void assign_descriptor_to_population(EA &ea, pop_t &pop, const RescaleFeature &prep) const {
                pop_t filtered_pop;
                for (auto ind:pop) {
                    if (!ind->fit().dead()) {
                        filtered_pop.push_back(ind);
                    } else {
                        std::vector<double> dd(Params::qd::behav_dim, -1.); // CHANGED from float to double
                        ind->fit().set_desc(dd);
                    }
                }

                Mat data;
                get_data(filtered_pop, data);
                Mat res; //will be resized
                get_descriptor_autoencoder(data, res, prep);

                for (size_t i = 0; i < filtered_pop.size(); i++) {
                    std::vector<double> dd;
                    for (size_t index_latent_space = 0;
                         index_latent_space < Params::qd::behav_dim;
                         ++index_latent_space) {

                        dd.push_back((double) res(i, index_latent_space));

                    }
                    filtered_pop[i]->fit().set_desc(dd);
                    filtered_pop[i]->fit().entropy() = (float) res(i, Params::qd::behav_dim);
                }

                if (!Params::nov::use_fixed_l) {
                    // Updating value for l

                    pop_t tmp_pop;
                    ea.container().get_full_content(tmp_pop);

                    if ((ea.gen() > 1) && (!tmp_pop.empty())) {
                        this->update_l(tmp_pop);
                    } else if (!tmp_pop.empty()){
                        this->initialise_l(tmp_pop);
                    }
                    std::cout << "l = " << Params::nov::l << "; size_pop = " << tmp_pop.size() << std::endl;

                }

            }

            template<typename EA>
            void assign_descriptor_to_population(EA &ea, pop_t &pop) const {
                assign_descriptor_to_population(ea, pop, _prep);
            }

            void get_data(const pop_t &pop, Mat &data) const {
//                // std::cout << "get_data" << std::endl;
//                if(pop[0]->fit().dead())
//
//                    std::cout << '\n'; // if no flush, then EIGEN (auto row=data.row(i)) gives an error

                data = Mat(pop.size(), pop[0]->fit().get_flat_obs_size());


                for (size_t i = 0; i < pop.size(); i++) {
                    // std::cout << data.rows() << std::endl;
                    auto row = data.row(i);
                    // std::cout << "here" << std::endl;
                    pop[i]->fit().get_flat_observations(row);
                }
                // std::cout << "get_data done" << std::endl;
            }

            void get_descriptor_autoencoder(const Mat &data, Mat &res, const RescaleFeature &prep) const {
                Mat scaled_data;
                prep.apply(data, scaled_data);
                Mat descriptor, entropy, loss, reconst;
                network->eval(scaled_data, descriptor, entropy, reconst);
                res = Mat(descriptor.rows(), descriptor.cols() + entropy.cols());
                res << descriptor, entropy;
            }


            stat_t get_stat(const pop_t &pop) {
                stat_t result;
                for (auto ind:pop)
                    result.push_back({ind->fit().desc(), ind->fit().value()});
                return result;
            }

            template<typename EA>
            void update_container(EA &ea) {
                pop_t tmp_pop;
                // Copy of the containt of the container into the _pop object.
                ea.container().get_full_content(tmp_pop);
                ea.container().erase_content();
                std::cout << "size pop: " << tmp_pop.size() << std::endl;

                this->assign_descriptor_to_population(ea, tmp_pop);

                //Update the population if the value of l has changed:F
                if (!Params::nov::use_fixed_l) {
                    // update l to maintain a number of indiv lower than 10k
                    std::cout << "NEW L= " << Params::nov::l << std::endl;

                    // Addition of the offspring to the container
                    std::vector<bool> added;
                    ea.add(tmp_pop, added);
                    ea.pop().clear();
                    // Copy of the content of the container into the _pop object.
                    ea.container().get_full_content(ea.pop());
                    // dump_data(ea,stat1,stat2,added);

                    std::cout << "Gen " << ea.gen() << " - size population with l updated : " << ea.pop().size() << std::endl;
                }
            }

            void update_l(const pop_t &pop) const {
                constexpr float alpha = 5e-6f;
                Params::nov::l *= (1 - alpha * (static_cast<float>(Params::resolution) - static_cast<float>(pop.size())));
            }

            void initialise_l(const pop_t &pop) const {
                Mat matrix_behavioural_descriptors;
                get_matrix_behavioural_descriptors(pop, matrix_behavioural_descriptors);
                Mat_dist abs_matrix{matrix_behavioural_descriptors.cast<double>()};

                abs_matrix = abs_matrix.rowwise() - abs_matrix.colwise().mean();

                Eigen::SelfAdjointEigenSolver<Mat_dist> eigensolver(abs_matrix.transpose() * abs_matrix);
                if (eigensolver.info() != 0) {
                    abort();
                }

                abs_matrix = (eigensolver.eigenvectors().transpose() * abs_matrix.transpose()).transpose();
                double volume = (abs_matrix.colwise().maxCoeff() - abs_matrix.colwise().minCoeff()).prod();

                Params::nov::l = static_cast<float>(0.5 * std::pow(volume / Params::resolution, 1. / matrix_behavioural_descriptors.cols()));
            }

            void get_matrix_behavioural_descriptors(const pop_t &pop, Mat &matrix_behavioural_descriptors) const {
                matrix_behavioural_descriptors = Mat(pop.size(), Params::qd::behav_dim);

                for (size_t i = 0; i < pop.size(); i++) {
                    auto desc = pop[i]->fit().desc();
                    for (size_t id = 0; id < Params::qd::behav_dim; id++) {
                        matrix_behavioural_descriptors(i, id) = desc[id];
                    }
                }
            }

            void distance(const Mat &X, Mat_dist &dist) const {
                // Compute norms
                Mat_dist X_double = X.cast<double>();
                Mat_dist XX = X_double.array().square().rowwise().sum();
                Mat_dist XY = (2 * X_double) * X_double.transpose();

                // Compute final expression
                dist = XX * Eigen::MatrixXd::Ones(1, XX.rows());
                dist = dist + Eigen::MatrixXd::Ones(XX.rows(), 1) * (XX.transpose());
                dist = dist - XY;
            }

            void get_reconstruction(const Mat &data, Mat &res) const {
                Mat scaled_data, scaled_res;
                _prep.apply(data, scaled_data);
                network->get_reconstruction(scaled_data, scaled_res);
                _prep.deapply(scaled_res, res);
            }

            NetworkLoader *get_network_loader() const {
                return &*network;
            }

            RescaleFeature& prep() {
                return _prep;
            }


        private:
            std::unique_ptr<NetworkLoader> network;
            RescaleFeature _prep;
            size_t last_update;
            size_t update_id;
        };
    }
}

#endif //SFERES2_DIMENSIONALITY_REDUCTION_HPP
