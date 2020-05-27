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

            // changing this to double gives an error downstream, as everything is in float
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

                if (Params::update::update_frequency == -1) 
                {// NOTE: THERE's BEEN A CHANGE to update_period
                    if (Params::update::update_period > 0 && 
                       (ea.gen() == 1 || ea.gen() == last_update + Params::update::update_period * std::pow(2, update_id - 1))) 
                    {
                        update_id++;
                        last_update = ea.gen();
                        update_descriptors(ea);
                    }
                } 
                else if (ea.gen() > 0) 
                {
                    if ((ea.gen() % Params::update::update_frequency == 0) || ea.gen() == 1) 
                    {
                        update_descriptors(ea);
                    }
                }

                if (!ea.offspring().size())
                {return;}
                assign_descriptor_to_population(ea, ea.offspring());
            }

            template<typename EA>
            void update_descriptors(EA &ea) {
                Mat phen_d, traj_d;
                std::vector<int> is_random_d;
                collect_dataset(phen_d, traj_d, is_random_d, ea, true); // gather the data from the indiv in the archive into a dataset
                // change train_netework
                train_network(phen_d, traj_d, is_random_d);
                update_container(ea);  // clear the archive and re-fill it using the new network
                // ea.pop_advers.clear(); // clearing adversarial examples
            }

            template<typename EA>
            void collect_dataset(Mat &phen_d, Mat &traj_d, std::vector<int> &is_random_d, EA &ea, bool training = false) const {
                std::vector<typename EA::indiv_t> content;
                
                // content will contain all phenotypes
                if (ea.gen() > 0) {
                    ea.container().get_full_content(content);
                } else {
                    content = ea.offspring();
                }

                // shuffle content here before getting the data so that the trajectories are also shuffled effectively
                if (training)
                {std::random_shuffle (content.begin(), content.end());}
                
                collect_dataset(phen_d, traj_d, is_random_d, ea, content, training);
            }

            template<typename EA>
            void collect_dataset(Mat &phen_d, Mat &traj_d, std::vector<int> &is_random_d,
                    EA &ea, const std::vector<typename EA::indiv_t>& content, bool training = false) const {
                
                // number of phenotypes
                size_t pop_size = content.size();
                get_phen(content, phen_d);
                get_trajectories(content, traj_d, is_random_d);
                
                if (training) {
                    std::cout << "Gen " << ea.gen() << " train set composed of " << phen_d.rows() << " phenotypes - Archive size : " << pop_size << std::endl;
                }
            }

            // test that this works
            void get_phen(const pop_t &pop, Mat &data) const {
                data = Mat(pop.size(), pop[0]->size());
                for (size_t i = 0; i < pop.size(); i++) {
                    for (size_t j{0}; j < pop[0]->size(); ++j)
                    {
                        data(i, j) = pop[i]->data(j);
                    }
                }
            }

            void get_trajectories(const pop_t &pop, Mat &data, std::vector<int> &is_random_d) const {
                data = Mat(pop.size() * (Params::random::max_num_random + 1), Params::sim::num_trajectory_elements);
                is_random_d.reserve(pop.size() * (Params::random::max_num_random + 1));
                
                size_t matrix_row_index{0};
                for (size_t i = 0; i < pop.size(); ++i) 
                {
                    // block of rows, populate the trajectories
                    auto block = data.block(matrix_row_index, 0, (Params::random::max_num_random + 1), Params::sim::num_trajectory_elements);
                    pop[i]->fit().get_flat_observations(block);
                    // populate the vector
                    for (size_t j {0}; j < Params::random::max_num_random + 1; ++j)
                    {
                        is_random_d.push_back(pop[i]->fit().is_random(j));
                    }
                    matrix_row_index += Params::random::max_num_random + 1;
                }
            }

            void train_network(const Mat &phen_d, const Mat &traj_d, std::vector<int> &is_random_d) {
                // we change the data normalisation each time we train/refine network, could cause small changes in loss between two trainings.
                // std::cout << "TRAINING" << std::endl;
                // std::cout << "ROWS" << phen_d.rows() << std::endl;
                _prep.init(phen_d);
                Mat scaled_data;
                _prep.apply(phen_d, scaled_data);
                // std::cout << "SCALED ROWS" << scaled_data.rows() << std::endl;
                float final_entropy = network->training(scaled_data, traj_d, is_random_d);
            }

            template<typename EA>
            void assign_descriptor_to_population(EA &ea, pop_t &pop) const {
                assign_descriptor_to_population(ea, pop, _prep);
            }

            template<typename EA>
            void assign_descriptor_to_population(EA &ea, pop_t &pop, const RescaleFeature &prep) const {
                pop_t filtered_pop;
                for (auto ind:pop) {
                    if (!ind->fit().dead()) 
                    {
                        filtered_pop.push_back(ind);
                    } 
                    // if dead
                    else 
                    {
                        std::vector<double> dd(Params::qd::behav_dim, -1.); // CHANGED from float to double
                        ind->fit().set_desc(dd);
                    }
                }

                Mat filtered_phen, filtered_traj;
                std::vector<int> is_trajectory;
                get_phen(filtered_pop, filtered_phen);
                get_trajectories(filtered_pop, filtered_traj, is_trajectory);

                // convert to eigen
                Eigen::VectorXi is_traj = Eigen::Map<Eigen::VectorXi> (is_trajectory.data(), is_trajectory.size());

                Mat latent_and_entropy;
                get_descriptor_autoencoder(filtered_phen, filtered_traj, is_traj, latent_and_entropy, prep);

                for (size_t i = 0; i < filtered_pop.size(); i++) {
                    std::vector<double> dd;
                    for (size_t index_latent_space = 0;
                         index_latent_space < Params::qd::behav_dim;
                         ++index_latent_space) {

                        dd.push_back((double) latent_and_entropy(i, index_latent_space));

                    }
                    filtered_pop[i]->fit().set_desc(dd);
                    filtered_pop[i]->fit().entropy() = (float) latent_and_entropy(i, Params::qd::behav_dim);
                }

                pop_t tmp_pop;
                ea.container().get_full_content(tmp_pop);

                if ((ea.gen() > 1) && (!tmp_pop.empty())) {
                    this->update_l(tmp_pop);
                } else if (!tmp_pop.empty()){
                    this->initialise_l(tmp_pop);
                }
                std::cout << "l = " << Params::nov::l << "; size_pop = " << tmp_pop.size() << std::endl;

            }

            void get_descriptor_autoencoder(const Mat &phen_d, const Mat &traj_d, const Eigen::VectorXi &is_trajectory, 
                                            Mat &latent_and_entropy, const RescaleFeature &prep) const {
                // _prep not initiialised again, uses the last one again?
                Mat scaled_data;
                prep.apply(phen_d, scaled_data);
                Mat descriptors, reconstructed_data, recon_loss;
                network->eval(phen_d, traj_d, is_trajectory, descriptors, reconstructed_data, recon_loss);
                latent_and_entropy = Mat(descriptors.rows(), descriptors.cols() + recon_loss.cols());
                latent_and_entropy << descriptors, recon_loss;
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
                // Copy of the content of the container into the _pop object.
                ea.container().get_full_content(tmp_pop);
                ea.container().erase_content();
                std::cout << "size pop: " << tmp_pop.size() << std::endl;

                this->assign_descriptor_to_population(ea, tmp_pop);

                // update l to maintain a number of indiv lower than Params::resolution
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

            void update_l(const pop_t &pop) const {
                constexpr float alpha = 5e-6f;
                Params::nov::l *= (1 - alpha * (static_cast<float>(Params::resolution) - static_cast<float>(pop.size())));
            }

            void get_matrix_behavioural_descriptors(const pop_t &pop, Mat &matrix_behavioural_descriptors) const 
            {
                matrix_behavioural_descriptors = Mat(pop.size(), Params::qd::behav_dim);

                for (size_t i = 0; i < pop.size(); i++) 
                {
                    auto desc = pop[i]->fit().desc();
                    for (size_t id = 0; id < Params::qd::behav_dim; id++) 
                    {
                        matrix_behavioural_descriptors(i, id) = desc[id];
                    }
                }
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

            void get_reconstruction(const Mat &phen, const Mat &traj, std::vector<int> &is_traj, Mat &reconstruction) const {
                Mat scaled_data;
                _prep.apply(phen, scaled_data);

                Eigen::VectorXi is_trajectories;
                get_network_loader()->vector_to_eigen(is_traj, is_trajectories);
                
                network->get_reconstruction(scaled_data, traj, is_trajectories, reconstruction);

                // do not need to apply scaling to reconstruction
                // _prep.deapply(scaled_reconstruction, reconstruction);
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
