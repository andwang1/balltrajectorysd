//
// Created by Luca Grillotti on 24/10/2019.
//

#ifndef SFERES2_DIMENSIONALITY_REDUCTION_HPP
#define SFERES2_DIMENSIONALITY_REDUCTION_HPP

#include <memory>
#include <sferes/stc.hpp>

namespace sferes {
    namespace modif {
        template<typename Phen, typename Params, typename NetworkLoader>
        class DimensionalityReduction {
        public:
            typedef Phen phen_t;
            typedef boost::shared_ptr<Phen> indiv_t;
            typedef typename std::vector<indiv_t> pop_t;
            typedef std::vector<std::pair<std::vector<double>, float>> stat_t;

            DimensionalityReduction() : _last_update(0), _update_id(0), _is_train_gen(false) {
                _network = std::make_unique<NetworkLoader>();
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

                _is_train_gen = false;
                if (Params::update::update_frequency == -1) 
                {
                    if (Params::update::update_period > 0 && 
                       (ea.gen() == 1 || ea.gen() == _last_update + Params::update::update_period * std::pow(2, _update_id - 1))) 
                    {
                        ++_update_id;
                        _last_update = ea.gen();
                        _is_train_gen = true;
                        update_descriptors(ea);
                    }
                } 
                else if (ea.gen() > 0) 
                {
                    if ((ea.gen() % Params::update::update_frequency == 0)) 
                    {
                        _is_train_gen = true;
                        update_descriptors(ea);
                    }
                }

                if (!ea.offspring().size())
                    {return;}
                assign_descriptor_to_population(ea, ea.offspring());
                
                if (Params::qd::num_train_archives > 0)
                    {refresh_l_train_archives(ea);}
            }

            template<typename EA>
            void update_descriptors(EA &ea) {
                Mat geno_d, traj_d;
                std::vector<int> is_trajectory;
                std::vector<typename EA::indiv_t> content;

                collect_dataset(geno_d, traj_d, is_trajectory, ea, content, true);

                #ifndef AURORA
                // add additionally the phenotypes that had the largest loss in the evaluation but with regenerated trajectories
                if (Params::ae::pct_extension > 0.001)
                {
                    Mat extended_geno, extended_traj;
                    std::vector<int> extended_is_traj;
                    extend_dataset(geno_d, traj_d, is_trajectory, ea, content, extended_geno, extended_traj, extended_is_traj, Params::ae::pct_extension);
                    train_network(extended_geno, extended_traj, extended_is_traj);
                }
                else
                    {train_network(geno_d, traj_d, is_trajectory);}

                #else // AURORA
                train_network(geno_d, traj_d, is_trajectory);
                #endif

                update_container(ea);  // clear the archive and re-fill it using the new network
                update_training_container(ea); // update all the training containers with new BDs
            }

            template<typename EA>
            void collect_dataset(Mat &geno_d, Mat &traj_d, std::vector<int> &is_trajectory, 
                                EA &ea, std::vector<typename EA::indiv_t> &content, bool training = false) const {
                // content will contain all phenotypes
                if (ea.gen() > 0) 
                    {ea.container().get_full_content(content);} 
                else 
                    {content = ea.offspring();}

                // get additional training content
                ea.get_full_content_train_archives(content);

                // shuffle content here before getting the data so that the trajectories are also shuffled effectively
                if (training)
                {std::random_shuffle (content.begin(), content.end());}
                
                // number of phenotypes
                size_t pop_size = content.size();
                get_geno(content, geno_d);
                get_trajectories(content, traj_d, is_trajectory);
                
                if (training) 
                {std::cout << "Gen " << ea.gen() << " train set composed of " << geno_d.rows() << " phenotypes - Archive size : " << pop_size << std::endl;}
            }

            template<typename EA>
            void extend_dataset(const Mat &geno_d, const Mat &traj_d, std::vector<int> &is_trajectory, EA &ea, std::vector<typename EA::indiv_t> &content, 
                                Mat &extended_geno, Mat &extended_traj, std::vector<int> &extended_is_traj, float pct_extension)
            {
                // extend dataset with the phenotypes with the pct_extension largest recon losses
                // phenotypes will be copied but trajectories will be regenerated, i.e. can have new random trajectories
                std::vector<typename EA::indiv_t> copied_pheno;
                get_additional_phenos(geno_d, traj_d, is_trajectory, content, ea, copied_pheno, pct_extension);

                // shuffle ahead of training
                std::random_shuffle (copied_pheno.begin(), copied_pheno.end());
                
                // retrieve the matrix data
                Mat additional_geno, additional_traj;
                std::vector<int> additional_is_traj;
                get_geno(copied_pheno, additional_geno);
                get_trajectories(copied_pheno, additional_traj, additional_is_traj);

                // put together for training
                extended_geno = Mat(geno_d.rows() + additional_geno.rows(), geno_d.cols());
                extended_traj = Mat(traj_d.rows() + additional_traj.rows(), traj_d.cols());

                // additional first so that they are included in train set after splitting
                extended_geno << additional_geno, geno_d;
                extended_traj << additional_traj, traj_d;

                extended_is_traj = additional_is_traj;
                extended_is_traj.reserve(is_trajectory.size() + additional_is_traj.size());
                extended_is_traj.insert(extended_is_traj.end(), is_trajectory.begin(), is_trajectory.end());
            }

            template<typename EA>
            void get_additional_phenos(const Mat &geno_d, const Mat &traj_d, std::vector<int> &is_trajectory, std::vector<typename EA::indiv_t> &content, EA &ea,
                                        std::vector<typename EA::indiv_t> &copied_pheno, float pct_extension)
            {
                Eigen::VectorXi is_trajectories;
                get_network_loader()->vector_to_eigen(is_trajectory, is_trajectories);

                Mat descriptors, reconstruction, recon_loss, recon_loss_unred, L2_loss, L2_loss_real_trajectories, KL_loss, encoder_var, decoder_var;
                _network->eval(geno_d, traj_d, is_trajectories, descriptors, reconstruction, recon_loss, recon_loss_unred, L2_loss, 
                               L2_loss_real_trajectories, KL_loss, encoder_var, decoder_var);

                int num_copies = pct_extension * geno_d.rows();
                copy_pheno(content, recon_loss, copied_pheno, num_copies, ea);
            }

            std::vector<size_t> sort_indexes(const Mat &vector) 
            {
                // initialize original index locations
                std::vector<size_t> idx(vector.size());
                std::iota(idx.begin(), idx.end(), 0);

                // sort indexes based on comparing values in v, std::stable_sort instead of std::sort to avoid unnecessary index re-orderings
                // when v contains elements of equal values 
                stable_sort(idx.begin(), idx.end(),
                    [&vector](size_t i1, size_t i2) {return vector(i1) > vector(i2);});
                return idx;
            }

            template<typename EA>
            void copy_pheno(const std::vector<typename EA::indiv_t> &content, const Mat &recon_loss, std::vector<typename EA::indiv_t> &copied_phen, 
                            size_t num_copies, EA &ea)
            {
                // sort phenotypes based on recon loss
                std::vector<size_t> sorted_indices = sort_indexes(recon_loss);
                
                // initialise new phenotypes
                copied_phen.resize(num_copies);
                BOOST_FOREACH (indiv_t& indiv, copied_phen) 
                {indiv = indiv_t(new Phen());}
                
                int count_has_random{0};
                // copy and generate new trajectories
                double total_dist_travelled{0};
                for (size_t i{0}; i < num_copies; ++i)
                {
                    *(copied_phen[i]) = *(content[sorted_indices[i]]);
                    copied_phen[i]->fit().eval(*copied_phen[i]);
                    float dist;
                    bool moved;
                    copied_phen[i]->fit().calculate_distance(dist, moved);
                    total_dist_travelled += dist;

                    // record the number of individuals that have a random trajectory attached
                    if (content[sorted_indices[i]]->fit().num_trajectories() > 0)
                    {++count_has_random;}
                }
                std::cout << "Additional Phen (hasrandom / total): " << count_has_random << "/" << num_copies << "\n";
                _random_extension_ratio = double(count_has_random) / num_copies;
                _avg_dist_travelled_random = total_dist_travelled / num_copies;
            }

            template<typename EA>
            void regenerate_pheno(const std::vector<typename EA::indiv_t> &content, std::vector<typename EA::indiv_t> &regenerated_phenos, 
                                  std::vector<size_t> &indices, EA &ea)
            {
                for (size_t i{0}; i < content.size(); ++i)
                {
                    // if has random trajectory
                    if (content[i]->fit().num_trajectories() > 0)
                    {
                        // record index
                        indices.push_back(i);

                        // create new phen and copy
                        regenerated_phenos.push_back(indiv_t(new Phen()));
                        *(regenerated_phenos[regenerated_phenos.size() - 1]) = *(content[i]);

                        // regenerate trajectory but without any random observations
                        (regenerated_phenos[regenerated_phenos.size() - 1])->fit().simulate(content[i]->fit().params());
                    }
                }
            }


            void get_geno(const pop_t &pop, Mat &data) const {
                data = Mat(pop.size(), pop[0]->gen().size());
                for (size_t i = 0; i < pop.size(); ++i) {
                    for (size_t j{0}; j < pop[0]->gen().size(); ++j)
                    {
                        data(i, j) = pop[i]->gen().data(j);
                    }
                }
            }

            void get_phen(const pop_t &pop, Mat &data) const {
                data = Mat(pop.size(), pop[0]->size());
                for (size_t i = 0; i < pop.size(); i++) {
                    for (size_t j{0}; j < pop[0]->size(); ++j)
                    {
                        data(i, j) = pop[i]->data(j);
                    }
                }
            }

            void get_trajectories(const pop_t &pop, Mat &data, std::vector<int> &is_trajectory) const {
                data = Mat(pop.size() * (Params::random::max_num_random + 1), Params::sim::num_trajectory_elements);
                is_trajectory.reserve(pop.size() * (Params::random::max_num_random + 1));
                
                size_t matrix_row_index{0};
                for (size_t i = 0; i < pop.size(); ++i) 
                {
                    // block of rows, populate the trajectories
                    auto block = data.block(matrix_row_index, 0, (Params::random::max_num_random + 1), Params::sim::num_trajectory_elements);
                    pop[i]->fit().get_flat_observations(block);
                    matrix_row_index += Params::random::max_num_random + 1;

                    // populate the vector
                    for (size_t j {0}; j < Params::random::max_num_random + 1; ++j)
                    {
                        is_trajectory.push_back(pop[i]->fit().is_random(j));
                    }
                    
                }
            }

            void get_stats(const Mat &geno, const Mat &traj, const Eigen::VectorXi &is_traj, 
                Mat &descriptors, Mat &reconstruction, Mat &recon_loss, Mat &recon_loss_unred,  
                Mat &L2_loss, Mat &L2_loss_real_trajectories, Mat &KL_loss, Mat &encoder_var, Mat &decoder_var) const
            {
                _network->eval(geno, traj, is_traj, descriptors, reconstruction, recon_loss, recon_loss_unred, 
                               L2_loss, L2_loss_real_trajectories, KL_loss, encoder_var, decoder_var);
            }

            void train_network(const Mat &geno_d, const Mat &traj_d, std::vector<int> &is_trajectory) {
                _network->training(geno_d, traj_d, is_trajectory);
            }

            template<typename EA>
            void assign_descriptor_to_population(EA &ea, pop_t &pop, int container_idx = 0) const {
                pop_t filtered_pop;
                for (auto ind:pop) {
                    if (!ind->fit().dead()) 
                        {filtered_pop.push_back(ind);} 
                    else // if dead
                    {
                        std::vector<double> dd(Params::qd::behav_dim, -1.); // CHANGED from float to double
                        ind->fit().set_desc(dd);
                    }
                }

                Mat filtered_geno, filtered_traj;
                std::vector<int> is_trajectory;
                get_geno(filtered_pop, filtered_geno);
                get_trajectories(filtered_pop, filtered_traj, is_trajectory);

                // convert to eigen
                Eigen::VectorXi is_traj = Eigen::Map<Eigen::VectorXi> (is_trajectory.data(), is_trajectory.size());

                Mat latent_and_entropy;
                get_descriptor_autoencoder(filtered_geno, filtered_traj, is_traj, latent_and_entropy);

                for (size_t i = 0; i < filtered_pop.size(); i++) 
                {
                    std::vector<double> dd;

                    for (size_t index_latent_space = 0; index_latent_space < Params::qd::behav_dim; ++index_latent_space) 
                    {dd.push_back((double) latent_and_entropy(i, index_latent_space));}

                    filtered_pop[i]->fit().set_desc(dd);
                    filtered_pop[i]->fit().entropy() = (float) latent_and_entropy(i, Params::qd::behav_dim);
                }

                if ((ea.gen() > 1) && (!filtered_pop.empty())) 
                {
                    if (container_idx == 0)
                        {this->update_l(ea.pop(), container_idx);}
                    else 
                        {this->update_l(filtered_pop, container_idx);}
                }
                else if (!filtered_pop.empty())
                    {this->initialise_l(filtered_pop, container_idx);}

                
                std::cout << "l = " << Params::nov::l[container_idx] << "; size_pop = ";
                if (container_idx == 0)
                    std::cout << ea.pop().size() << std::endl;
                else
                    {std::cout << filtered_pop.size() << std::endl;}

            }

            template<typename EA>
            void refresh_l_train_archives(EA &ea) const 
            {
                for (int i{0}; i < Params::qd::num_train_archives; ++i)
                {
                    if ((ea.gen() > 1) && (ea.train_container(i).archive().size() != 0) && (Params::nov::l[i + 1] != 0)) 
                        {this->update_l(ea.train_container(i).archive().size(), i + 1);} 
                    else if (ea.train_container(i).archive().size() != 0)
                    {   
                        pop_t tmp_pop;
                        ea.train_container(i).get_full_content(tmp_pop);
                        this->initialise_l(tmp_pop, i + 1);
                    }
                }
            }

            void get_descriptor_autoencoder(const Mat &geno_d, const Mat &traj_d, const Eigen::VectorXi &is_trajectory, 
                                            Mat &latent_and_entropy) const 
            {
                Mat descriptors, reconstructed_data, recon_loss, recon_loss_unred, L2_loss, L2_loss_real_trajectories, KL_loss, encoder_var, decoder_var;
                _network->eval(geno_d, traj_d, is_trajectory, descriptors, reconstructed_data, recon_loss, recon_loss_unred, 
                               L2_loss, L2_loss_real_trajectories, KL_loss, encoder_var, decoder_var);

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
                std::cout << "NEW L= " << Params::nov::l[0] << std::endl;

                // Addition of the offspring to the container
                std::vector<bool> added;
                ea.add(tmp_pop, added);
                ea.pop().clear();
                // Copy of the content of the container into the _pop object.
                ea.container().get_full_content(ea.pop());
                // dump_data(ea,stat1,stat2,added);

                std::cout << "Gen " << ea.gen() << " - size population with l updated : " << ea.pop().size() << std::endl;
            }

            template<typename EA>
            void update_training_container(EA &ea) {
                for (int i{0}; i < Params::qd::num_train_archives; ++i)
                {
                    if (ea.train_container(i).archive().size() == 0)
                        continue;
                        
                    pop_t tmp_pop;
                    // Copy of the content of the container into the _pop object.
                    ea.train_container(i).get_full_content(tmp_pop);
                    ea.train_container(i).erase_content();
                    std::cout << "Container " << i << " size pop: " << tmp_pop.size() << std::endl;

                    this->assign_descriptor_to_population(ea, tmp_pop, i + 1);

                    // update l to maintain a number of indiv lower than Params::resolution
                    std::cout << "NEW L= " << Params::nov::l[i + 1] << std::endl;
                    
                    // Put data back into the container
                    std::vector<bool> added(tmp_pop.size());
                    for (size_t j{0}; j < tmp_pop.size(); ++j)
                    {added[j] = ea.train_container(i).add(tmp_pop[j], i + 1);}

                    pop_t empty;
                    ea.train_container(i).update(tmp_pop, empty);

                    std::cout << "Gen " << ea.gen() << " - container "<< i << " size after BD update : " << ea.train_container(i).archive().size() << std::endl;
                }
            }

            void update_l(const pop_t &pop, size_t container_idx) const {
                constexpr float alpha = 5e-6f;
                Params::nov::l[container_idx] *= (1 - alpha * (static_cast<float>(Params::qd::resolution) - static_cast<float>(pop.size())));
            }

            void update_l(size_t pop_size, size_t container_idx) const {
                constexpr float alpha = 5e-6f;
                Params::nov::l[container_idx] *= (1 - alpha * (static_cast<float>(Params::qd::resolution) - static_cast<float>(pop_size)));
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

            void initialise_l(const pop_t &pop, int container_idx) const {
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

                Params::nov::l[container_idx] = static_cast<float>(0.5 * std::pow(volume / Params::qd::resolution, 1. / matrix_behavioural_descriptors.cols()));
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

            double get_random_extension_ratio() const
            {return _random_extension_ratio;}

            double get_extension_avg_dist_travelled() const
            {return _avg_dist_travelled_random;}

            bool is_train_gen() const
            {return _is_train_gen;}

            NetworkLoader *get_network_loader() const {
                return &*_network;
            }

        private:
            std::unique_ptr<NetworkLoader> _network;
            size_t _last_update;
            size_t _update_id;
            double _random_extension_ratio{-1};
            double _avg_dist_travelled_random{-1};
            bool _is_train_gen;
        };
    }
}

#endif //SFERES2_DIMENSIONALITY_REDUCTION_HPP
