#ifndef __NETWORK__LOADER__HPP__
#define __NETWORK__LOADER__HPP__

#include <sferes/misc/rand.hpp>

#include <chrono>
#include <iomanip>
#include <tuple>

#include "autoencoder/encoder.hpp"
#include "autoencoder/decoder.hpp"
#include "autoencoder/autoencoder.hpp"

template <typename TParams, typename Exact = stc::Itself>
class AbstractLoader : public stc::Any<Exact> {
public:
    typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXf_rm;

    explicit AbstractLoader(std::size_t latent_size, torch::nn::AnyModule auto_encoder_module) :
            m_global_step(0),
            m_auto_encoder_module(std::move(auto_encoder_module)),
            m_adam_optimiser(torch::optim::Adam(m_auto_encoder_module.ptr()->parameters(),
                                                torch::optim::AdamOptions(TParams::ae::learning_rate)
                                                        .beta1(0.5))),
            m_device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU)
        {

        if (torch::cuda::is_available()) {
            std::cout << "Torch -> Using CUDA" << std::endl;
        } else {
            std::cout << "Torch -> Using CPU" << std::endl;
        }

        this->m_auto_encoder_module.ptr()->to(this->m_device);
    }

    void eval(const MatrixXf_rm &phen,
              const MatrixXf_rm &traj,
              const Eigen::VectorXi &is_traj,
              MatrixXf_rm &descriptors,
              MatrixXf_rm &reconstructed_data,
              MatrixXf_rm &recon_loss) {
        stc::exact(this)->eval(phen, traj, is_traj, descriptors, reconstructed_data, recon_loss);
    }
    
    void prepare_batches(std::vector<std::tuple<torch::Tensor, torch::Tensor, std::vector<bool>>> &batches, 
                        const MatrixXf_rm &phen, const MatrixXf_rm &traj, const Eigen::VectorXi &is_trajectory) const {
        stc::exact(this)->prepare_batches(batches, phen, traj, is_trajectory);
    }

    size_t split_dataset(const MatrixXf_rm &phen_d, const MatrixXf_rm &traj_d,
                       MatrixXf_rm &train_phen, MatrixXf_rm &valid_phen, 
                       MatrixXf_rm &train_traj, MatrixXf_rm &valid_traj) 
        {
        size_t l_train_phen{0}, l_valid_phen{0}, l_train_traj{0}, l_valid_traj{0};
        
        if (phen_d.rows() > 500) 
        {
            l_train_phen = floor(phen_d.rows() * Options::CV_fraction);
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

        // loop through all trajectories, filter out bad ones and note where the boundaries lie
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
        MatrixXf_rm train_phen, valid_phen, train_traj, valid_traj;
        size_t l_train_traj = this->split_dataset(phen_d, traj_d, train_phen, valid_phen, train_traj, valid_traj);
        // split the bool vector according to the same split
        std::vector<int> train_is_trajectories(is_trajectories.begin(), is_trajectories.begin() + l_train_traj);
        std::vector<int> val_is_trajectories(is_trajectories.begin() + l_train_traj, is_trajectories.end());

        if (train_traj.rows() == valid_traj.rows())
        {
            val_is_trajectories = train_is_trajectories;
        }

        // is this needed?
        // train_is_trajectories.resize(l_train_traj);
        // val_is_trajectories.resize(is_trajectories.size() - l_train_traj);

        // change vectors to eigen
        Eigen::VectorXi tr_is_traj, val_is_traj, is_traj;
        vector_to_eigen(train_is_trajectories, tr_is_traj);
        vector_to_eigen(val_is_trajectories, val_is_traj);
        vector_to_eigen(is_trajectories, is_traj);
        
        // std::cout << "BEFORE AVG" << std::endl;
        // std::cout << train_phen.rows() << std::endl;
        // std::cout << valid_phen.rows() << std::endl;
        // std::cout << train_traj.rows() << std::endl;
        // std::cout << valid_traj.rows() << std::endl;
        // std::cout << tr_is_traj.size() << std::endl;
        // std::cout << val_is_traj.size() << std::endl;


        float init_tr_recon_loss = this->get_avg_recon_loss(train_phen, train_traj, tr_is_traj);
        float init_vl_recon_loss = this->get_avg_recon_loss(valid_phen, valid_traj, val_is_traj);

        std::cout << "INIT recon train loss: " << init_tr_recon_loss << "   valid recon loss: " << init_vl_recon_loss << std::endl;
        std::cout << "Training: Total num of trajectories " << is_traj.sum() << ", Num random trajectories " << is_traj.sum() - phen_d.rows() << ", (random ratio: " << 1 - float(phen_d.rows())/is_traj.sum() <<")" << std::endl;
        bool _continue = true;
        Eigen::VectorXd previous_avg = Eigen::VectorXd::Ones(5) * 100;

        int nb_epochs = Options::nb_epochs;

        int epoch(0);

        while (_continue && (epoch < nb_epochs)) {
            std::vector<std::tuple<torch::Tensor, torch::Tensor, std::vector<bool>>> batches;
            prepare_batches(batches, train_phen, train_traj, tr_is_traj);

            for (auto &tup : batches) {
                // Get the names below with the inspect_graph.py script applied on the generated graph_text.pb file.
                this->m_auto_encoder_module.ptr()->zero_grad();
                // tup[0] is the phenotype
                torch::Tensor reconstruction_tensor = this->m_auto_encoder_module.forward(std::get<0>(tup));
                torch::Tensor loss_tensor = torch::zeros(1);;
                // start at -1 because first loop will take it to 0
                int index{-1};

                // std::vector<bool> &boundaries = std::get<2>(tup);

                // tup[2] is the boundaries vector
                for (int i{0}; i < std::get<2>(tup).size(); ++i)
                {
                    if (std::get<2>(tup)[i])
                    {++index;}
                    // second arg is type of norm, here L2, third argument is which dimensions to sum over
                    // tup[1] is the trajectories tensor
                    // std::cout << "BEFORE NORM" << std::endl;
                    // std::cout << "recon" << reconstruction_tensor[index] << std::endl;
                    // std::cout << "traj" << std::get<1>(tup)[i] << std::endl;
                    loss_tensor += torch::norm(std::get<1>(tup)[i] - reconstruction_tensor[index], 2, {0});
                }
                long num_trajectories {std::get<2>(tup).size()};
                loss_tensor /= num_trajectories;
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

                if ((previous_avg.array() - previous_avg[0]).mean() > 0 &&
                    this->get_avg_recon_loss(train_phen, train_traj, tr_is_traj) < init_tr_recon_loss)
                    _continue = false;
            }


            float recon_loss_t = this->get_avg_recon_loss(train_phen, train_traj, tr_is_traj);
            float recon_loss_v = this->get_avg_recon_loss(valid_phen, valid_traj, val_is_traj);


            std::cout.precision(5);
            std::cout << "training dataset: " << train_phen.rows() << "  valid dataset: " << valid_phen.rows() << " - ";
            std::cout << std::setw(5) << epoch << "/" << std::setw(5) << nb_epochs;
            std::cout << " recon loss (t): " << std::setw(8) << recon_loss_t;
            std::cout << " (v): " << std::setw(8) << recon_loss_v;
            std::cout << std::flush << '\r';
        }

        float full_dataset_recon_loss = this->get_avg_recon_loss(phen_d, traj_d, is_traj);
        std::cout << "Final full dataset recon loss: " << full_dataset_recon_loss << '\n';

        return full_dataset_recon_loss;
    }

    void get_reconstruction(const MatrixXf_rm &phen, const MatrixXf_rm &traj, const Eigen::VectorXi &is_traj, 
                            MatrixXf_rm &reconstruction) {
        MatrixXf_rm descriptors, recon_loss;
        eval(phen, traj, is_traj, descriptors, reconstruction, recon_loss);
        std::cout << "AFTER GET RECON" << std::endl;
    }

    float get_avg_recon_loss(const MatrixXf_rm &phen, const MatrixXf_rm &traj, const Eigen::VectorXi &is_traj) {
        MatrixXf_rm descriptors, reconst, recon_loss;
        eval(phen, traj, is_traj, descriptors, reconst, recon_loss);
        return recon_loss.mean();
    }

    void vector_to_eigen(std::vector<int> &is_trajectories, Eigen::VectorXi &is_traj)
    {
        is_traj = Eigen::Map<Eigen::VectorXi> (is_trajectories.data(), is_trajectories.size());
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

    struct Options {
        // config setting
        static const size_t batch_size = TParams::ae::batch_size;
        static const size_t nb_epochs = TParams::ae::nb_epochs;
        static constexpr float convergence_epsilon = TParams::ae::convergence_epsilon;
        SFERES_CONST float CV_fraction = TParams::ae::CV_fraction;
    };

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
};

template <typename TParams, typename Exact = stc::Itself>
class NetworkLoaderAutoEncoder : public AbstractLoader<TParams, typename stc::FindExact<NetworkLoaderAutoEncoder<TParams, Exact>, Exact>::ret> {
public:
    typedef AbstractLoader<TParams, typename stc::FindExact<NetworkLoaderAutoEncoder<TParams, Exact>, Exact>::ret> TParentLoader;

    explicit NetworkLoaderAutoEncoder() :
            TParentLoader(TParams::qd::behav_dim,
                          torch::nn::AnyModule(AutoEncoder(TParams::qd::gen_dim, TParams::ae::en_hid_dim1, TParams::qd::behav_dim, 
                                                           TParams::ae::de_hid_dim1, TParams::ae::de_hid_dim2, TParams::sim::num_trajectory_elements)))
             {}

    typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXf_rm;

    void prepare_batches(std::vector<std::tuple<torch::Tensor, torch::Tensor, std::vector<bool>>> &batches, 
                        const MatrixXf_rm &phen, const MatrixXf_rm &traj, const Eigen::VectorXi &is_trajectory) const 
    {
        if (phen.rows() <= TParentLoader::Options::batch_size) {
            batches = std::vector<std::tuple<torch::Tensor, torch::Tensor, std::vector<bool>>>(1);
        } else {
            batches = std::vector<std::tuple<torch::Tensor, torch::Tensor, std::vector<bool>>>(floor(phen.rows() / (TParentLoader::Options::batch_size)));
        }

        // in loop do filtering before passing to make tuple
        if (batches.size() == 1) 
        {
                // filtering
                MatrixXf_rm filtered_traj;
                std::vector<bool> boundaries;
                this->filter_trajectories(traj, is_trajectory,filtered_traj, boundaries);

                torch::Tensor T1, T2;
                this->get_tuple_from_eigen_matrices(phen, filtered_traj, boundaries, T1, T2, batches[0]);
        } 
        else 
        {
            for (size_t ind = 0; ind < batches.size(); ind++) 
            {
                // filtering
                MatrixXf_rm filtered_traj;
                std::vector<bool> boundaries;
                this->filter_trajectories(traj.middleRows(ind * TParentLoader::Options::batch_size * (TParams::random::max_num_random + 1),
                                                    TParentLoader::Options::batch_size * (TParams::random::max_num_random + 1)),
                                                    is_trajectory.segment(ind * TParentLoader::Options::batch_size * (TParams::random::max_num_random + 1), TParentLoader::Options::batch_size * (TParams::random::max_num_random + 1)),
                                                    filtered_traj,
                                                    boundaries);

                torch::Tensor T1, T2;
                this->get_tuple_from_eigen_matrices(phen.middleRows(ind * TParentLoader::Options::batch_size, TParentLoader::Options::batch_size),
                                                    filtered_traj,
                                                    boundaries,
                                                    T1,
                                                    T2,
                                                    batches[ind]);
            }
        }

    }

    void eval(const MatrixXf_rm &phen,
              const MatrixXf_rm &traj,
              const Eigen::VectorXi &is_trajectory,
              MatrixXf_rm &descriptors,
              MatrixXf_rm &reconstructed_data,
              MatrixXf_rm &recon_loss) 
    {
        AutoEncoder auto_encoder = std::static_pointer_cast<AutoEncoderImpl>(this->m_auto_encoder_module.ptr());

        torch::Tensor phen_tensor, traj_tensor;
        this->get_torch_tensor_from_eigen_matrix(phen, phen_tensor);
        
        MatrixXf_rm filtered_traj;
        std::vector<bool> boundaries;
        this->filter_trajectories(traj,
                        is_trajectory,
                        filtered_traj,
                        boundaries);

        this->get_torch_tensor_from_eigen_matrix(filtered_traj, traj_tensor);

        torch::Tensor descriptors_tensor;
        torch::Tensor reconstruction_tensor = auto_encoder->forward_get_latent(phen_tensor, descriptors_tensor);
        torch::Tensor reconstruction_loss = torch::zeros(phen.rows());

        // start at -1 because first loop will take it to 0
        int index{-1};
        for (int i{0}; i < boundaries.size(); ++i)
        {
            if (boundaries[i])
            {++index;}
            // second arg is type of norm, here L2, third argument is which dimensions to sum over
            // dim = {0} because the difference between the two is automatically squeezed from 2D to 1D, row by row difference
            reconstruction_loss[index] += torch::norm(traj_tensor[i] - reconstruction_tensor[index], 2, {0});
        }

        this->get_eigen_matrix_from_torch_tensor(descriptors_tensor.cpu(), descriptors);
        this->get_eigen_matrix_from_torch_tensor(reconstruction_tensor.cpu(), reconstructed_data);
        this->get_eigen_matrix_from_torch_tensor(reconstruction_loss.cpu(), recon_loss);
        // std::cout << "Eval: Total num of trajectories " << boundaries.size() << ", Num random trajectories " << boundaries.size() - phen.rows() << ", (random ratio: " << 1 - float(phen.rows())/boundaries.size() <<")" << std::endl;
    }
};

#endif
