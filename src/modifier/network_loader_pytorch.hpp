#ifndef __NETWORK__LOADER__HPP__
#define __NETWORK__LOADER__HPP__

#include <sferes/misc/rand.hpp>

#include <chrono>
#include <iomanip>
#include <tuple>


#ifdef VAE
#include "autoencoder/autoencoder_VAE.hpp"
#include "autoencoder/encoder_VAE.hpp"
#include "autoencoder/decoder_VAE.hpp"
#else
#include "autoencoder/autoencoder_AE.hpp"
#include "autoencoder/encoder_AE.hpp"
#include "autoencoder/decoder_AE.hpp"
#endif

template <typename TParams, typename Exact = stc::Itself>
class AbstractLoader : public stc::Any<Exact> {
public:
    typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXf_rm;

    explicit AbstractLoader(std::size_t latent_size, torch::nn::AnyModule auto_encoder_module) :
            m_global_step(0),
            m_auto_encoder_module(std::move(auto_encoder_module)),
            m_adam_optimiser(torch::optim::Adam(m_auto_encoder_module.ptr()->parameters(),
                                                torch::optim::AdamOptions(TParams::ae::learning_rate)
                                                        .betas(std::make_tuple(0.9, 0.999)))),
            m_device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU)
    {
        if (torch::cuda::is_available()) 
        {
            const char* cuda_visible_devices = std::getenv("CUDA_VISIBLE_DEVICES");
            std::string str_cuda_visible_devices = (cuda_visible_devices == NULL) ? std::string("") : std::string(cuda_visible_devices);
            if (not str_cuda_visible_devices.empty()) 
            {
                int index_device_to_use = std::stoi(str_cuda_visible_devices);
                m_device = torch::Device(m_device.type(), index_device_to_use);
                std::cout << "Torch -> Using CUDA ; index device: " << index_device_to_use << std::endl;
            } 
            else 
                {std::cout << "Torch -> Using CUDA ; no specified index device " << std::endl;}

        } 
        else {std::cout << "Torch -> Using CPU" << std::endl;}

        this->m_auto_encoder_module.ptr()->to(this->m_device);
    }

    void eval(const MatrixXf_rm &phen,
              const MatrixXf_rm &traj,
              const Eigen::VectorXi &is_traj,
              MatrixXf_rm &descriptors,
              MatrixXf_rm &reconstructed_data,
              MatrixXf_rm &recon_loss,
              MatrixXf_rm &recon_loss_unred,
              MatrixXf_rm &L2_loss,
              MatrixXf_rm &L2_loss_real_trajectories,
              MatrixXf_rm &KL_loss,
              MatrixXf_rm &encoder_var,
              MatrixXf_rm &decoder_var) {
        stc::exact(this)->eval(phen, traj, is_traj, descriptors, reconstructed_data, recon_loss, recon_loss_unred, 
                               L2_loss, L2_loss_real_trajectories, KL_loss, encoder_var, decoder_var);
    }
    
    void prepare_batches(std::vector<std::tuple<torch::Tensor, torch::Tensor, std::vector<bool>>> &batches, 
                        const MatrixXf_rm &phen, const MatrixXf_rm &traj, const Eigen::VectorXi &is_trajectory) const {
        stc::exact(this)->prepare_batches(batches, phen, traj, is_trajectory);
    }

    size_t split_dataset(const MatrixXf_rm &phen_d, const MatrixXf_rm &traj_d,
                       MatrixXf_rm &train_phen, MatrixXf_rm &valid_phen, 
                       MatrixXf_rm &train_traj, MatrixXf_rm &valid_traj) 
    {
        size_t l_train_phen, l_valid_phen, l_train_traj, l_valid_traj;
        
        if (phen_d.rows() > 500) 
        {
            l_train_phen = floor(phen_d.rows() * TParams::ae::CV_fraction);
            l_valid_phen = phen_d.rows() - l_train_phen;
            
            // every phen has multiple trajectories
            l_train_traj = l_train_phen * (TParams::random::max_num_random + 1);
            l_valid_traj = traj_d.rows() - l_train_traj;
        } 
        else 
        {
            l_train_phen = phen_d.rows();
            l_valid_phen = phen_d.rows();

            l_train_traj = l_train_phen * (TParams::random::max_num_random + 1);
            l_valid_traj = l_train_traj;
        }
        assert(l_train_phen != 0 && l_valid_phen != 0);

        train_phen = phen_d.topRows(l_train_phen);
        valid_phen = phen_d.bottomRows(l_valid_phen);

        train_traj = traj_d.topRows(l_train_traj);
        valid_traj = traj_d.bottomRows(l_valid_traj);

        return l_train_traj;
    }

    void filter_trajectories(const MatrixXf_rm &trajectories, const Eigen::VectorXi &is_trajectory, 
                            MatrixXf_rm &filtered_trajectories, std::vector<bool> &boundaries) const
    {
        int num_actual_trajectories = is_trajectory.sum();

        filtered_trajectories = MatrixXf_rm(num_actual_trajectories, trajectories.cols());
        
        int filtered_row_index{0};

        // loop through all trajectories, filter out trajectories that are not actual trajectories and note where the boundaries lie between phenotypes
        for (int i{0}; i < is_trajectory.size(); ++i)
        {
            if (is_trajectory(i) == 1)
            {
                filtered_trajectories.row(filtered_row_index) = trajectories.row(i);
                ++filtered_row_index;

                // marking which trajectories belong to which phenotype, the first trajectory is the boundary marker
                if (i % (TParams::random::max_num_random + 1) == 0)
                    {boundaries.push_back(true);}
                else
                    {boundaries.push_back(false);}
            }
        }
    }

    float training(const MatrixXf_rm &phen_d, const MatrixXf_rm &traj_d, std::vector<int> &is_trajectories, bool full_train = false, int generation = 1000) 
    {
        return stc::exact(this)->training(phen_d, traj_d, is_trajectories, full_train, generation);
    }

    float get_avg_recon_loss(const MatrixXf_rm &phen, const MatrixXf_rm &traj, const Eigen::VectorXi &is_traj) {
        MatrixXf_rm descriptors, reconst, recon_loss, recon_loss_unred, L2_loss, L2_loss_real_trajectories, KL_loss, encoder_var, decoder_var;
        eval(phen, traj, is_traj, descriptors, reconst, recon_loss, recon_loss_unred, L2_loss, L2_loss_real_trajectories, KL_loss, encoder_var, decoder_var);
        return recon_loss.mean();
    }

    void get_torch_tensor_from_eigen_matrix(const MatrixXf_rm &M, torch::Tensor &T) const {
        T = torch::rand({M.rows(), M.cols()});
        float *data = T.data_ptr<float>();
        memcpy(data, M.data(), M.cols() * M.rows() * sizeof(float));
    }

    void get_eigen_matrix_from_torch_tensor(const torch::Tensor &T, MatrixXf_rm &M) const {
        if (T.dim() == 0) {
            M = MatrixXf_rm(1, 1); //scalar
            float *data = T.data_ptr<float>();
            M = Eigen::Map<MatrixXf_rm>(data, 1, 1);
        } else {
            size_t total_size_individual_tensor = 1;
            for (size_t dim = 1; dim < T.dim(); ++dim) {
                total_size_individual_tensor *= T.size(dim);
            }
            M = MatrixXf_rm(T.size(0), total_size_individual_tensor);
            float *data = T.data_ptr<float>();
            M = Eigen::Map<MatrixXf_rm>(data, T.size(0), total_size_individual_tensor);
        }
    }

    void get_tuple_from_eigen_matrices(const MatrixXf_rm &M1, const MatrixXf_rm &M2, const std::vector<bool> &boundaries,
                                        torch::Tensor &T1, torch::Tensor &T2, 
                                        std::tuple<torch::Tensor, torch::Tensor, std::vector<bool>> &tuple) const {
        T1 = torch::rand({M1.rows(), M1.cols()});
        T2 = torch::rand({M2.rows(), M2.cols()});
        get_torch_tensor_from_eigen_matrix(M1, T1);
        get_torch_tensor_from_eigen_matrix(M2, T2);
        tuple = std::make_tuple(T1, T2, boundaries);
    }

    torch::nn::AnyModule get_auto_encoder() {
        return this->m_auto_encoder_module;
    }

    torch::nn::AnyModule& auto_encoder() {
        return this->m_auto_encoder_module;
    }

    int32_t m_global_step;

protected:
    torch::nn::AnyModule m_auto_encoder_module;
    torch::optim::Adam m_adam_optimiser;
    torch::Device m_device;
    double _log_2_pi;
};

template <typename TParams, typename Exact = stc::Itself>
class NetworkLoaderAutoEncoder : public AbstractLoader<TParams, typename stc::FindExact<NetworkLoaderAutoEncoder<TParams, Exact>, Exact>::ret> {
public:
    typedef AbstractLoader<TParams, typename stc::FindExact<NetworkLoaderAutoEncoder<TParams, Exact>, Exact>::ret> TParentLoader;

    explicit NetworkLoaderAutoEncoder() :
            TParentLoader(TParams::qd::behav_dim,
                          torch::nn::AnyModule(AutoEncoder(TParams::qd::gen_dim, TParams::ae::en_hid_dim1, TParams::ae::en_hid_dim2, TParams::qd::behav_dim, 
                                                           TParams::ae::de_hid_dim1, TParams::ae::de_hid_dim2, TParams::sim::num_trajectory_elements, TParams::ae::bias))),
            _log_2_pi(log(2 * M_PI)),
            _epochs_trained(0) {}

    typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXf_rm;

    void prepare_batches(std::vector<std::tuple<torch::Tensor, torch::Tensor, std::vector<bool>>> &batches, 
                        const MatrixXf_rm &phen, const MatrixXf_rm &traj, const Eigen::VectorXi &is_trajectory) const 
    {
        if (phen.rows() <= TParams::ae::batch_size) 
            {batches = std::vector<std::tuple<torch::Tensor, torch::Tensor, std::vector<bool>>>(1);} 
        else 
            {batches = std::vector<std::tuple<torch::Tensor, torch::Tensor, std::vector<bool>>>(floor(phen.rows() / (TParams::ae::batch_size)));}

        // in loop do filtering before passing to make tuple
        if (batches.size() == 1) 
        {
            // filtering
            MatrixXf_rm filtered_traj, scaled_filtered_traj;
            std::vector<bool> boundaries;
            this->filter_trajectories(traj, is_trajectory, filtered_traj, boundaries);

            torch::Tensor T1, T2;
            this->get_tuple_from_eigen_matrices(phen, filtered_traj, boundaries, T1, T2, batches[0]);
        } 
        else 
        {
            for (size_t ind = 0; ind < batches.size(); ++ind) 
            {
                // filtering
                MatrixXf_rm filtered_traj, scaled_filtered_traj;
                std::vector<bool> boundaries;
                this->filter_trajectories(traj.middleRows(ind * TParams::ae::batch_size * (TParams::random::max_num_random + 1),
                                                    TParams::ae::batch_size * (TParams::random::max_num_random + 1)),
                                                    is_trajectory.segment(ind * TParams::ae::batch_size * (TParams::random::max_num_random + 1), TParams::ae::batch_size * (TParams::random::max_num_random + 1)),
                                                    filtered_traj, boundaries);

                torch::Tensor T1, T2;
                this->get_tuple_from_eigen_matrices(phen.middleRows(ind * TParams::ae::batch_size, TParams::ae::batch_size),
                                                    filtered_traj, boundaries,
                                                    T1, T2, batches[ind]);
            }
        }
    }

    void vector_to_eigen(std::vector<int> &is_trajectories, Eigen::VectorXi &is_traj) const
    {is_traj = Eigen::Map<Eigen::VectorXi> (is_trajectories.data(), is_trajectories.size());}

    int get_epochs_trained() const
    {return _epochs_trained;}

    void get_sq_dist_matrix(const torch::Tensor &data, torch::Tensor &dist_mat) const
    {
        // batch size by batch size shape
        dist_mat = torch::empty({data.size(0), data.size(0)}, torch::device(this->m_device));
        torch::Tensor scalar_one = torch::ones(1, torch::dtype(torch::kLong));
        for (int i{0}; i < data.size(0); ++i)
            // subtraction broadcasts over columns
            {dist_mat.index_put_({scalar_one * i}, torch::sum(torch::pow(data - data.index({i}), 2), {1}), false);}
    }

    // binary search to find variances
    float get_ith_var_from_perplexity(const torch::Tensor &dist_mat_row, long num_row) const
    {
        float tolerance = 1e-5;

        // batch size * 0.1 as per VAE-SNE
        float target_perplexity = dist_mat_row.size(0) * 0.1;

        torch::Tensor scalar_two = torch::ones({1}, torch::device(this->m_device)) * 2;
        
        // for masking out the log(0) term
        torch::Tensor mask = torch::arange(dist_mat_row.size(0), torch::dtype(torch::kLong));
        
        // init var at 1
        torch::Tensor var = torch::ones(1, torch::device(this->m_device));

        float min_var{0};
        float max_var = FLT_MAX;

        int iter{0};
        while (iter < 100)
        {
            // std::cout << "Var Search Iter: " << iter << " Current Var.: " << var.item<float>() << std::endl;

            // nominator of p_j|i
            torch::Tensor exp_cur_sim_mat_row = torch::exp(-dist_mat_row / var);
            // need to mask out the ith term in the summation, subtract 1 as the distance term will be 0, so in the exp matrix, e^0 = 1
            torch::Tensor p_j_i = exp_cur_sim_mat_row / (torch::sum(exp_cur_sim_mat_row) - 1);
            
            // set p_i_i to 0
            p_j_i.index_put_({num_row}, 0);
            // std::cout << "CE: " << torch::sum((p_j_i * torch::log2(p_j_i + 1e-14)).index(mask.ne(num_row))) << std::endl;
            float cur_perplexity = torch::pow(scalar_two, -torch::sum((p_j_i * torch::log2(p_j_i + 1e-14)).index(mask.ne(num_row)))).item<float>();

            // std::cout << "Current Perplexity" << cur_perplexity << std::endl;
            // std::cout << "Current difference " << cur_perplexity - target_perplexity << std::endl;

            if (abs(target_perplexity - cur_perplexity) < tolerance)
                {break;}
            
            // if perplexity too high, then need to decrease variance
            else if (cur_perplexity > target_perplexity)
            {
                max_var = var.item<float>();
                var = (var + min_var) / 2;
            }
            else
            {
                min_var = var.item<float>();
                if (max_var == FLT_MAX)
                {var = 2 * var;}
                else
                {var = (var + max_var) / 2;}
            }
            // std::cout << "MAX: " << max_var << " MIN: " << min_var << std::endl;
            ++iter;
        }
        // std::cout << "Iters needed: " << iter << ", Final Var. " << var.item<float>() << "\n";
        return var.item<float>();
    }

    void get_var_from_perplexity(const torch::Tensor &dist_mat, torch::Tensor &variances) const
    {
        variances = torch::empty(dist_mat.size(0), torch::device(this->m_device));
        tbb::parallel_for(tbb::blocked_range<long>(0, dist_mat.size(0)),
            [&](tbb::blocked_range<long> r)
        {
            for (long i=r.begin(); i<r.end(); ++i)
                {variances.index_put_({i}, get_ith_var_from_perplexity(dist_mat.index({i}), i));}
        });
    }

    float training(const MatrixXf_rm &phen_d, const MatrixXf_rm &traj_d, std::vector<int> &is_trajectories, bool full_train = false, int generation = 1000) 
    {
        AutoEncoder auto_encoder = std::static_pointer_cast<AutoEncoderImpl>(this->m_auto_encoder_module.ptr());
        auto_encoder->train();
        std::cout << "Total Size Dataset incl. readditions: " << phen_d.rows() << std::endl;
        MatrixXf_rm train_phen, valid_phen, train_traj, valid_traj;
        size_t l_train_traj = this->split_dataset(phen_d, traj_d, train_phen, valid_phen, train_traj, valid_traj);
        // split the bool vector according to the same split
        std::vector<int> train_is_trajectories(is_trajectories.begin(), is_trajectories.begin() + l_train_traj);
        std::vector<int> val_is_trajectories(is_trajectories.begin() + l_train_traj, is_trajectories.end());

        // if the dataset is too small -> see split_dataset
        if (train_traj.rows() == valid_traj.rows())
        {val_is_trajectories = train_is_trajectories;}

        // change vectors to eigen
        Eigen::VectorXi tr_is_traj, val_is_traj, is_traj;
        vector_to_eigen(train_is_trajectories, tr_is_traj);
        vector_to_eigen(val_is_trajectories, val_is_traj);
        vector_to_eigen(is_trajectories, is_traj);
        
        float init_tr_recon_loss = this->get_avg_recon_loss(train_phen, train_traj, tr_is_traj);
        float init_vl_recon_loss = this->get_avg_recon_loss(valid_phen, valid_traj, val_is_traj);

        std::cout << "INIT recon train loss: " << init_tr_recon_loss << "   valid recon loss: " << init_vl_recon_loss << std::endl;
        std::cout << "Training: Total num of trajectories " << is_traj.sum() << ", Num random trajectories " << is_traj.sum() - phen_d.rows() << ", (random ratio: " << 1 - float(phen_d.rows())/is_traj.sum() <<")" << std::endl;
        bool _continue = true;
        Eigen::VectorXd previous_avg = Eigen::VectorXd::Ones(TParams::ae::running_mean_num_epochs) * 50;

        int epoch(0);

        // initialise training variables
        torch::Tensor encoder_mu, encoder_logvar, decoder_logvar, descriptors_tensor;

        while (_continue && (epoch < TParams::ae::nb_epochs)) {
            std::vector<std::tuple<torch::Tensor, torch::Tensor, std::vector<bool>>> batches;
            prepare_batches(batches, train_phen, train_traj, tr_is_traj);

            for (auto &tup : batches) {
                // Get the names below with the inspect_graph.py script applied on the generated graph_text.pb file.
                this->m_auto_encoder_module.ptr()->zero_grad();
                
                // not necessary as layers enforce grad
                // std::get<0>(tup).set_requires_grad(true);

                // tup[1] is the trajectories tensor
		        torch::Tensor traj = std::get<1>(tup).to(this->m_device);

                // tup[0] is the phenotype
                torch::Tensor reconstruction_tensor = auto_encoder->forward_get_latent(std::get<0>(tup).to(this->m_device), encoder_mu, encoder_logvar, decoder_logvar, 
                                                                                descriptors_tensor, TParams::qd::sample);
                torch::Tensor loss_tensor = torch::zeros(1, torch::device(this->m_device));

                // start at -1 because first loop will take it to 0
                int index{-1};

                // tup[2] is the boundaries vector
                for (int i{0}; i < std::get<2>(tup).size(); ++i)
                {
                    if (std::get<2>(tup)[i])
                    {++index;}

                    if (TParams::ae::full_loss)
                    {
                        if (TParams::ae::loss_function == TParams::ae::loss::L2)
                            {loss_tensor += torch::sum(torch::pow(traj[i] - reconstruction_tensor[index], 2) / (2 * torch::exp(decoder_logvar[index])) + 0.5 * (decoder_logvar[index] + _log_2_pi));}
                        else if (TParams::ae::loss_function == TParams::ae::loss::L1)
                            {loss_tensor += torch::sum(torch::abs(traj[i] - reconstruction_tensor[index]) / (2 * torch::exp(decoder_logvar[index])) + 0.5 * (decoder_logvar[index] + _log_2_pi));}
                        else if (TParams::ae::loss_function == TParams::ae::loss::SmoothL1)
                            {loss_tensor += torch::sum(torch::smooth_l1_loss(reconstruction_tensor[index], traj[i], 0) / (2 * torch::exp(decoder_logvar[index])) + 0.5 * (decoder_logvar[index] + _log_2_pi));}
                    }
                    else
                    {
                        if (TParams::ae::loss_function == TParams::ae::loss::L2)
                            {loss_tensor += torch::sum(torch::pow(traj[i] - reconstruction_tensor[index], 2));}
                        else if (TParams::ae::loss_function == TParams::ae::loss::L1)
                            {loss_tensor += torch::sum(torch::abs(traj[i] - reconstruction_tensor[index]));}
                        else if (TParams::ae::loss_function == TParams::ae::loss::SmoothL1)
                            {loss_tensor += torch::sum(torch::smooth_l1_loss(reconstruction_tensor[index], traj[i], 0));}
                    }
                }

                long num_trajectories {static_cast<long>(std::get<2>(tup).size())};
                loss_tensor /= num_trajectories;

                #ifdef VAE
                // SNE / TSNE
                // get the high dimensional similarities
                torch::Tensor h_dist_mat, h_variances;
                get_sq_dist_matrix(reconstruction_tensor.detach(), h_dist_mat);
                get_var_from_perplexity(h_dist_mat, h_variances);

                // similarity matrix, unsqueeze so division is along columns
                torch::Tensor exp_h_sim_mat = torch::exp(-h_dist_mat / h_variances.unsqueeze(1));

                // here need to mask out the index i as per TSNE paper (not proper KL factor 1: p_j_i not summing to 1)
                torch::Tensor p_j_i = exp_h_sim_mat / (torch::sum(exp_h_sim_mat, {1}) - 1).unsqueeze(1);

                // set diagonal to zero as only interested in pairwise similarities, as per TSNE paper (not proper KL factor 2, not summing to 1)
                p_j_i.fill_diagonal_(0);

                // get the low dimensional similarities
                torch::Tensor l_dist_mat;
                get_sq_dist_matrix(descriptors_tensor, l_dist_mat);
                if (TParams::ae::TSNE)
                {
                    // not proper KL factor 3: dividing by 2n
                    torch::Tensor p_ij = (p_j_i + p_j_i.transpose(0, 1)) / (2 * p_j_i.size(0));
                    
                    torch::Tensor l_sim_mat = 1 / (1 + l_dist_mat);

                    // here need to mask out the index i as per TSNE paper, ith term will be = 1 as dist = 0
                    torch::Tensor q_ij = l_sim_mat / (torch::sum(l_sim_mat, {1}) - 1).unsqueeze(1);
                    // set diagonal to zero as only interested in pairwise similarities, as per TSNE paper
                    q_ij.fill_diagonal_(0);

                    // torch::Tensor tsne = p_ij * torch::log(p_ij / q_ij);
                    // the above equation is proportional to the below, since the p values are constants wrt the derivative that we are taking
                    torch::Tensor tsne = -p_ij * torch::log(q_ij);

                    // set 0 * log(0) terms to 0
                    tsne.fill_diagonal_(0);
                    
                    // set coefficient to dimensionality of data as per VAE-SNE paper
                    loss_tensor += torch::sum(tsne) * reconstruction_tensor.size(1) / reconstruction_tensor.size(0);
                }
                else // SNE
                {
                    torch::Tensor exp_l_sim_mat = torch::exp(-l_dist_mat);

                    // here need to mask out the index i as per the paper
                    torch::Tensor q_ij = exp_l_sim_mat / (torch::sum(exp_l_sim_mat, {1}) - 1).unsqueeze(1);
                    // set diagonal to zero as only interested in pairwise similarities, as per TSNE paper
                    q_ij.fill_diagonal_(0);

                    // torch::Tensor sne = p_j_i * torch::log(p_j_i / q_ij);
                    // the above equation is proportional to the below, since the p values are constants wrt the derivative that we are taking
                    torch::Tensor sne = -p_j_i * torch::log(q_ij);
    
                    // set 0 * log(0) terms to 0
                    sne.fill_diagonal_(0);

                    // set coefficient to dimensionality of data as per VAE-SNE paper
                    loss_tensor += torch::sum(sne) * reconstruction_tensor.size(1) / reconstruction_tensor.size(0);
                }

                // KL Loss
                loss_tensor += -0.5 * TParams::ae::beta * torch::sum(1 + encoder_logvar - torch::pow(encoder_mu, 2) - torch::exp(encoder_logvar), {1}).mean();
                #endif
                loss_tensor.backward();
                this->m_adam_optimiser.step();
                ++epoch;
            }

            this->m_global_step++;

            // early stopping
            if (!full_train) {
                float current_avg = this->get_avg_recon_loss(valid_phen, valid_traj, val_is_traj);
                for (size_t t = 1; t < previous_avg.size(); t++)
                    previous_avg[t - 1] = previous_avg[t];

                previous_avg[previous_avg.size() - 1] = current_avg;

                // if the running average on the val set is increasing and train loss is higher than at the beginning
                if ((previous_avg.array() - previous_avg[0]).mean() > 0 && epoch > TParams::ae::min_num_epochs &&
                    this->get_avg_recon_loss(train_phen, train_traj, tr_is_traj) < init_tr_recon_loss)
                        {_continue = false;}
            }

            float recon_loss_t = this->get_avg_recon_loss(train_phen, train_traj, tr_is_traj);
            float recon_loss_v = this->get_avg_recon_loss(valid_phen, valid_traj, val_is_traj);

            std::cout.precision(5);
            std::cout << "training dataset: " << train_phen.rows() << "  valid dataset: " << valid_phen.rows() << " - ";
            std::cout << std::setw(5) << epoch << "/" << std::setw(5) << TParams::ae::nb_epochs;
            std::cout << " recon loss (t): " << std::setw(8) << recon_loss_t;
            std::cout << " (v): " << std::setw(8) << recon_loss_v;
            std::cout << std::flush << '\r';
        }

        float full_dataset_recon_loss = this->get_avg_recon_loss(phen_d, traj_d, is_traj);
        std::cout << "Final full dataset recon loss: " << full_dataset_recon_loss << '\n';

        _epochs_trained = epoch + 1;

        return full_dataset_recon_loss;
    }

    void eval(const MatrixXf_rm &phen,
              const MatrixXf_rm &traj,
              const Eigen::VectorXi &is_trajectory,
              MatrixXf_rm &descriptors,
              MatrixXf_rm &reconstructed_data,
              MatrixXf_rm &recon_loss,
              MatrixXf_rm &recon_loss_unred,
              MatrixXf_rm &L2_loss,
              MatrixXf_rm &L2_loss_real_trajectories,
              MatrixXf_rm &KL_loss,
              MatrixXf_rm &encoder_var,
              MatrixXf_rm &decoder_var) 
    {
        torch::NoGradGuard no_grad;
        AutoEncoder auto_encoder = std::static_pointer_cast<AutoEncoderImpl>(this->m_auto_encoder_module.ptr());
        auto_encoder->eval();

        torch::Tensor phen_tensor, traj_tensor;
        this->get_torch_tensor_from_eigen_matrix(phen, phen_tensor);
        
        MatrixXf_rm filtered_traj, scaled_filtered_traj;
        std::vector<bool> boundaries;
        this->filter_trajectories(traj, is_trajectory, filtered_traj, boundaries);
        this->get_torch_tensor_from_eigen_matrix(filtered_traj, traj_tensor);
	    traj_tensor = traj_tensor.to(this->m_device);

        torch::Tensor encoder_mu, encoder_logvar, decoder_logvar, descriptors_tensor;
        torch::Tensor reconstruction_tensor = auto_encoder->forward_get_latent(phen_tensor.to(this->m_device), encoder_mu, encoder_logvar, decoder_logvar, 
                                                                                descriptors_tensor, TParams::qd::sample);
        torch::Tensor reconstruction_loss = torch::zeros(phen.rows(), torch::device(this->m_device));
        
        
        #ifdef VAE
        // KL divergence
        torch::Tensor KL = -0.5 * TParams::ae::beta * (1 + encoder_logvar - torch::pow(encoder_mu, 2) - torch::exp(encoder_logvar));
        #endif

        torch::Tensor L2 = torch::empty({filtered_traj.rows(), TParams::sim::num_trajectory_elements}, torch::device(this->m_device));
        torch::Tensor L2_actual_traj = torch::empty(phen.rows(), torch::device(this->m_device));
        torch::Tensor recon_loss_unreduced = torch::empty_like(L2, torch::device(this->m_device));

        int index{0};
        int internal_avg_counter{0};

        for (int i{0}; i < boundaries.size(); ++i)
        {
            if (boundaries[i] && i > 0)
            {
                reconstruction_loss[index] /= internal_avg_counter;
                ++index;
                internal_avg_counter = 0;
            }

            #ifdef VAE
            if (TParams::ae::full_loss)
            {
                if (TParams::ae::loss_function == TParams::ae::loss::L2)
                    {recon_loss_unreduced[i] = torch::pow(traj_tensor[i] - reconstruction_tensor[index], 2) / (2 * torch::exp(decoder_logvar[index])) + 0.5 * (decoder_logvar[index] + _log_2_pi);}
                else if (TParams::ae::loss_function == TParams::ae::loss::L1)
                    {recon_loss_unreduced[i] = torch::abs(traj_tensor[i] - reconstruction_tensor[index]) / (2 * torch::exp(decoder_logvar[index])) + 0.5 * (decoder_logvar[index] + _log_2_pi);}
                else if (TParams::ae::loss_function == TParams::ae::loss::SmoothL1)
                    {recon_loss_unreduced[i] = torch::smooth_l1_loss(reconstruction_tensor[index], traj_tensor[i], 0) / (2 * torch::exp(decoder_logvar[index])) + 0.5 * (decoder_logvar[index] + _log_2_pi);}
                reconstruction_loss[index] += torch::sum(recon_loss_unreduced[i]) + torch::sum(KL[index]);
            }
            else
            {
                if (TParams::ae::loss_function == TParams::ae::loss::L2)
                    {recon_loss_unreduced[i] = torch::pow(traj_tensor[i] - reconstruction_tensor[index], 2);}
                else if (TParams::ae::loss_function == TParams::ae::loss::L1)
                    {recon_loss_unreduced[i] = torch::abs(traj_tensor[i] - reconstruction_tensor[index]);}
                else if (TParams::ae::loss_function == TParams::ae::loss::SmoothL1)
                    {recon_loss_unreduced[i] = torch::smooth_l1_loss(reconstruction_tensor[index], traj_tensor[i], 0);}
                reconstruction_loss[index] += torch::sum(recon_loss_unreduced[i]) + torch::sum(KL[index]);
            }
            L2[i] = torch::pow(traj_tensor[i] - reconstruction_tensor[index], 2);
            
            if (boundaries[i])
            L2_actual_traj[index] = torch::sum(L2[i]);

            #else //AE
            if (TParams::ae::loss_function == TParams::ae::loss::L2)
                {recon_loss_unreduced[i] = torch::pow(traj_tensor[i] - reconstruction_tensor[index], 2);}
            else if (TParams::ae::loss_function == TParams::ae::loss::L1)
                {recon_loss_unreduced[i] = torch::abs(traj_tensor[i] - reconstruction_tensor[index]);}
            else if (TParams::ae::loss_function == TParams::ae::loss::SmoothL1)                
                {recon_loss_unreduced[i] = torch::smooth_l1_loss(reconstruction_tensor[index], traj_tensor[i], 0);}
            reconstruction_loss[index] += torch::sum(recon_loss_unreduced[i]);
            
            if (boundaries[i])
                L2_actual_traj[index] = torch::sum(torch::pow(traj_tensor[i] - reconstruction_tensor[index], 2));
            #endif

            ++internal_avg_counter;
        }
        // outside loop, if last one is a random trajectory, then the last element in boundaries is false, and thus need to divide
        if (!boundaries[boundaries.size() - 1])
        {reconstruction_loss[index] /= internal_avg_counter;}

        MatrixXf_rm scaled_reconstructed_data;

        this->get_eigen_matrix_from_torch_tensor(descriptors_tensor.cpu(), descriptors);
        this->get_eigen_matrix_from_torch_tensor(reconstruction_tensor.cpu(), reconstructed_data);
        this->get_eigen_matrix_from_torch_tensor(reconstruction_loss.cpu(), recon_loss);
        this->get_eigen_matrix_from_torch_tensor(recon_loss_unreduced.cpu(), recon_loss_unred);
        this->get_eigen_matrix_from_torch_tensor(L2_actual_traj.cpu(), L2_loss_real_trajectories);

        #ifdef VAE
        this->get_eigen_matrix_from_torch_tensor(torch::exp(decoder_logvar).cpu(), decoder_var);
        this->get_eigen_matrix_from_torch_tensor(torch::exp(encoder_logvar).cpu(), encoder_var);
        this->get_eigen_matrix_from_torch_tensor(L2.cpu(), L2_loss);
        this->get_eigen_matrix_from_torch_tensor(KL.cpu(), KL_loss);
        #endif
    }

    float _log_2_pi;
    int _epochs_trained;
};

#endif
