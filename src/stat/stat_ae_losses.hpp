//
// Created by Andy Wang
//

#ifndef SFERES2_STAT_AE_LOSSES_HPP
#define SFERES2_STAT_AE_LOSSES_HPP

#include <sferes/stat/stat.hpp>

namespace sferes {
    namespace stat {

        SFERES_STAT(Losses, Stat)
        {
        public:
            typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> matrix_t;

            template<typename EA>
            void refresh(EA &ea) 
            {
                std::string prefix = "ae_loss";
                _write_losses(prefix, ea);
            }
            

            template<typename EA>
            void _write_losses(const std::string &prefix, const EA &ea) {

                std::string fname = ea.res_dir() + "/" + prefix + std::string(".dat");
                std::cout << "writing... " << fname << std::endl;

                // retrieve all phenotypes and trajectories                
                matrix_t gen, traj;
                std::vector<int> is_traj;
                boost::fusion::at_c<0>(ea.fit_modifier()).get_geno(ea.pop(), gen);
                boost::fusion::at_c<0>(ea.fit_modifier()).get_trajectories(ea.pop(), traj, is_traj);
                
                Eigen::VectorXi is_trajectory;
                boost::fusion::at_c<0>(ea.fit_modifier()).get_network_loader()->vector_to_eigen(is_traj, is_trajectory);

                matrix_t descriptors, recon_loss, recon_loss_unred, reconstruction, L2_loss, L2_loss_real_trajectories, KL_loss, encoder_var, decoder_var;
                boost::fusion::at_c<0>(ea.fit_modifier()).get_stats(gen, traj, is_trajectory, descriptors, reconstruction, recon_loss, recon_loss_unred, 
                                                                    L2_loss, L2_loss_real_trajectories, KL_loss, encoder_var, decoder_var);

                std::ofstream ofs(fname.c_str(), std::ofstream::app);
                ofs.precision(17);
                double recon = recon_loss.mean();
                double L2_real_traj = L2_loss_real_trajectories.mean();
                
                #ifdef VAE
                float sne_loss;
                if (boost::fusion::at_c<0>(ea.fit_modifier()).is_train_gen())
                {
                    // TSNE loss
                    torch::Tensor reconstruction_tensor, descriptors_tensor;
                    boost::fusion::at_c<0>(ea.fit_modifier()).get_network_loader()->get_torch_tensor_from_eigen_matrix(reconstruction, reconstruction_tensor);
                    boost::fusion::at_c<0>(ea.fit_modifier()).get_network_loader()->get_torch_tensor_from_eigen_matrix(descriptors, descriptors_tensor);

                    if (torch::cuda::is_available())
                    {
                        reconstruction_tensor = reconstruction_tensor.to(torch::device(torch::kCUDA));
                        descriptors_tensor = descriptors_tensor.to(torch::device(torch::kCUDA));
                    }

                    // get the high dimensional similarities
                    torch::Tensor h_dist_mat, h_variances;
                    boost::fusion::at_c<0>(ea.fit_modifier()).get_network_loader()->get_sq_dist_matrix(reconstruction_tensor, h_dist_mat);
                    boost::fusion::at_c<0>(ea.fit_modifier()).get_network_loader()->get_var_from_perplexity(h_dist_mat, h_variances);

                    // similarity matrix, unsqueeze so division is along columns
                    torch::Tensor exp_h_sim_mat = torch::exp(-h_dist_mat / h_variances.unsqueeze(1));

                    // here need to mask out the index i as per TSNE paper (not proper KL factor 1: not summing to 1)
                    torch::Tensor p_j_i = exp_h_sim_mat / (torch::sum(exp_h_sim_mat, {1}) - 1).unsqueeze(1);

                    // set diagonal to zero as only interested in pairwise similarities, as per TSNE paper (not proper KL factor 2, on top of factor 1, not summing to 1)
                    p_j_i.fill_diagonal_(0);

                    // get the low dimensional similarities
                    torch::Tensor l_dist_mat;
                    boost::fusion::at_c<0>(ea.fit_modifier()).get_network_loader()->get_sq_dist_matrix(descriptors_tensor, l_dist_mat);

                    if (Params::ae::TSNE)
                    {
                        torch::Tensor p_ij = (p_j_i + p_j_i.transpose(0, 1)) / (2 * p_j_i.size(0));
                        
                        torch::Tensor l_sim_mat = 1 / (1 + l_dist_mat);

                        // here need to mask out the index i as per TSNE paper, ith term will be = 1 as dist = 0, so = e^1
                        torch::Tensor q_ij = l_sim_mat / (torch::sum(l_sim_mat, {1}) - torch::exp(torch::ones(1))).unsqueeze(1);
                        
                        // set diagonal to zero as only interested in pairwise similarities, as per TSNE paper
                        q_ij.fill_diagonal_(0);

                        torch::Tensor tsne = p_ij * torch::log(p_ij / q_ij);

                        // set 0 * log(0) terms to 0
                        tsne.fill_diagonal_(0);
                        // set coefficient to dimensionality of data as per VAE-SNE paper
                        sne_loss = (torch::sum(tsne) * reconstruction_tensor.size(1) / reconstruction_tensor.size(0)).item<float>();
                    }
                    else // SNE
                    {
                        torch::Tensor exp_l_sim_mat = torch::exp(-l_dist_mat);

                        // here need to mask out the index i as per the paper
                        torch::Tensor q_ij = exp_l_sim_mat / (torch::sum(exp_l_sim_mat, {1}) - 1).unsqueeze(1);
                        // set diagonal to zero as only interested in pairwise similarities, as per TSNE paper
                        q_ij.fill_diagonal_(0);

                        torch::Tensor sne = p_j_i * torch::log(p_j_i / q_ij);
        
                        // set 0 * log(0) terms to 0
                        sne.fill_diagonal_(0);
                        // set coefficient to dimensionality of data as per VAE-SNE paper
                        sne_loss = (torch::sum(sne) * reconstruction_tensor.size(1) / reconstruction_tensor.size(0)).item<float>();
                    }
                }
                // these three are unreduced, need row wise sum and then mean
                float L2 = L2_loss.rowwise().sum().mean();
                float KL = KL_loss.rowwise().sum().mean();
                float de_var = decoder_var.rowwise().sum().mean();
                float en_var = encoder_var.rowwise().sum().mean();
                
                // retrieve trajectories without any interference from random observations
                matrix_t undisturbed_traj(ea.pop().size(), Params::sim::num_trajectory_elements);
                for (size_t i{0}; i < ea.pop().size(); ++i)
                {undisturbed_traj.row(i) = ea.pop()[i]->fit().get_undisturbed_trajectory();}
                float L2_undisturbed_real_traj = (undisturbed_traj - reconstruction).array().square().rowwise().sum().mean();

                ofs << ea.gen() << ", " << recon << ", " << L2 << ", " << KL << ", " << en_var << ", " << de_var << ", " << ", " << L2_real_traj << ", " << L2_undisturbed_real_traj;
                #else

                #ifdef AURORA
                ofs << ea.gen() << ", " << recon << ", " << L2_real_traj;
                #else // AE

                matrix_t undisturbed_traj(ea.pop().size(), Params::sim::num_trajectory_elements);
                for (size_t i{0}; i < ea.pop().size(); ++i)
                {undisturbed_traj.row(i) = ea.pop()[i]->fit().get_undisturbed_trajectory();}
                float L2_undisturbed_real_traj = (undisturbed_traj - reconstruction).array().square().rowwise().sum().mean();

                ofs << ea.gen() << ", " << recon << ", " << L2_real_traj << ", " << L2_undisturbed_real_traj;
                #endif 

                #endif

                if (boost::fusion::at_c<0>(ea.fit_modifier()).is_train_gen())
                {
                    #ifdef VAE
                    ofs << ", " << sne_loss;
                    #endif

                    #ifndef AURORA
                    ofs << ", " << boost::fusion::at_c<0>(ea.fit_modifier()).get_random_extension_ratio() << ", " << boost::fusion::at_c<0>(ea.fit_modifier()).get_network_loader()->get_epochs_trained() << "/" << Params::ae::nb_epochs << ", IS_TRAIN";
                    #else
                    ofs << ", " << boost::fusion::at_c<0>(ea.fit_modifier()).get_network_loader()->get_epochs_trained() << "/" << Params::ae::nb_epochs << ", IS_TRAIN";
                    #endif
                }
                ofs << "\n";
            }
        };

    }
}


#endif
