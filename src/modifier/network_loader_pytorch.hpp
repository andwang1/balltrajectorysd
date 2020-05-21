#ifndef __NETWORK__LOADER__HPP__
#define __NETWORK__LOADER__HPP__

#include <torch/torch.h>

#include <sferes/misc/rand.hpp>

#include <chrono>
#include <iomanip>

#include "autoencoder/encoder.hpp"
#include "autoencoder/decoder.hpp"
#include "autoencoder/autoencoder.hpp"
#include "autoencoder/lstm_auto_encoder.hpp"

template <typename TParams, typename Exact = stc::Itself>
class AbstractLoader : public stc::Any<Exact> {
public:
    typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXf_rm;

    explicit AbstractLoader(std::size_t latent_size, torch::nn::AnyModule auto_encoder_module) :
            m_global_step(0),
            m_auto_encoder_module(std::move(auto_encoder_module)),
            m_adam_optimiser(torch::optim::Adam(m_auto_encoder_module.ptr()->parameters(),
                                                torch::optim::AdamOptions(2e-4)
                                                        .beta1(0.5))),
            m_device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU){

        if (torch::cuda::is_available()) {
            std::cout << "Torch -> Using CUDA" << std::endl;
        } else {
            std::cout << "Torch -> Using CPU" << std::endl;
        }

        this->m_auto_encoder_module.ptr()->to(this->m_device);
    }

    void eval(const MatrixXf_rm &data,
              MatrixXf_rm &descriptors,
              MatrixXf_rm &recon_loss,
              MatrixXf_rm &reconstructed_data) {
        stc::exact(this)->eval(data, descriptors, recon_loss, reconstructed_data);
    }

    void prepare_batches(std::vector<torch::Tensor> &batches, const MatrixXf_rm &data) const {
        stc::exact(this)->prepare_batches(batches, data);
    }

    void split_dataset(const MatrixXf_rm &data, MatrixXf_rm &train, MatrixXf_rm &valid) {
        Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> perm(data.rows());
        perm.setIdentity();
        std::random_shuffle(perm.indices().data(), perm.indices().data() + perm.indices().size());
        MatrixXf_rm tmp = perm * data;
        size_t l_train{0}, l_valid{0};
        if (data.rows() > 500) {
            l_train = floor(data.rows() * Options::CV_fraction);
            l_valid = data.rows() - l_train;
        } else {
            l_train = data.rows();
            l_valid = data.rows();
        }
        assert(l_train != 0 && l_valid != 0);

        train = tmp.topRows(l_train);
        valid = tmp.bottomRows(l_valid);
    }

    float training(const MatrixXf_rm &data, bool full_train = false, int generation = 1000) {
        MatrixXf_rm train_db, valid_db;
        this->split_dataset(data, train_db, valid_db);

        float init_tr_recon_loss = this->get_avg_recon_loss(train_db);
        float init_vl_recon_loss = this->get_avg_recon_loss(valid_db);

        std::cout << "INIT recon train loss: " << init_tr_recon_loss << "   valid recon loss: " << init_vl_recon_loss;

        bool _continue = true;
        Eigen::VectorXd previous_avg = Eigen::VectorXd::Ones(5) * 100;

        int nb_epochs = Options::nb_epochs;

        int epoch(0);

        while (_continue && (epoch < nb_epochs)) {
            this->split_dataset(data, train_db, valid_db);

            std::vector<torch::Tensor> batches;
            prepare_batches(batches, train_db);

            for (auto &batche : batches) {
                // Get the names below with the inspect_graph.py script applied on the generated graph_text.pb file.
                this->m_auto_encoder_module.ptr()->zero_grad();
                torch::Tensor reconstruction_tensor = this->m_auto_encoder_module.forward(batche);
                torch::Tensor loss_reconstruction = torch::mse_loss(reconstruction_tensor, batche);
                loss_reconstruction.backward();
                this->m_adam_optimiser.step();
                ++epoch;
            }

            this->m_global_step++;


            // early stopping
            if (!full_train) {
                float current_avg = this->get_avg_recon_loss(valid_db);
                for (size_t t = 1; t < previous_avg.size(); t++)
                    previous_avg[t - 1] = previous_avg[t];

                previous_avg[previous_avg.size() - 1] = current_avg;

                if ((previous_avg.array() - previous_avg[0]).mean() > 0 &&
                    this->get_avg_recon_loss(train_db) < init_tr_recon_loss)
                    _continue = false;
            }


            float recon_loss_t = this->get_avg_recon_loss(train_db);
            float recon_loss_v = this->get_avg_recon_loss(valid_db);


            std::cout.precision(5);
            std::cout << "training dataset: " << train_db.rows() << "  valid dataset: " << valid_db.rows() << " - ";
            std::cout << std::setw(5) << epoch << "/" << std::setw(5) << nb_epochs;
            std::cout << " recon loss (t): " << std::setw(8) << recon_loss_t;
            std::cout << " (v): " << std::setw(8) << recon_loss_v;
            std::cout << std::flush << '\r';
        }
        std::cout << "Final recon loss: " << this->get_avg_recon_loss(data) << '\n';

        return this->get_avg_recon_loss(data);
    }

    void get_reconstruction(const MatrixXf_rm &data, MatrixXf_rm &reconstruction) {
        MatrixXf_rm desc, recon_loss;
        eval(data, desc, recon_loss, reconstruction);
    }


    float get_avg_recon_loss(const MatrixXf_rm &data) {
        MatrixXf_rm descriptors, recon_loss, reconst;
        eval(data, descriptors, recon_loss, reconst);
        return recon_loss.mean();
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
        static const int input_dim = 2;
        static const int batch_size = 20000;
        static const int nb_epochs = 10000;
        static constexpr float convergence_epsilon = 0.0000001;
        SFERES_CONST float CV_fraction = 0.75;
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
};

template <typename TParams, typename Exact = stc::Itself>
class NetworkLoaderAutoEncoder : public AbstractLoader<TParams, typename stc::FindExact<NetworkLoaderAutoEncoder<TParams, Exact>, Exact>::ret> {
public:
    typedef AbstractLoader<TParams, typename stc::FindExact<NetworkLoaderAutoEncoder<TParams, Exact>, Exact>::ret> TParentLoader;

    explicit NetworkLoaderAutoEncoder() :
            TParentLoader(TParams::qd::behav_dim,
                          torch::nn::AnyModule(AutoEncoder(32, 32, TParams::qd::behav_dim, TParams::use_colors))),
            m_use_colors(TParams::use_colors) {

        if (this->m_use_colors) {
            std::cout << "Using COLORS" << std::endl;
        } else {
            std::cout << "Using GRAYSCALE Images" << std::endl;
        }
    }

    typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXf_rm;


    void prepare_batches(std::vector<torch::Tensor> &batches, const MatrixXf_rm &data) const {
        /*
         * Generate all the batches for training
         * */
        if (data.rows() <= TParentLoader::Options::batch_size) {
            batches = std::vector<torch::Tensor>(1);
        } else {
            batches = std::vector<torch::Tensor>(floor(data.rows() / (TParentLoader::Options::batch_size)));
        }

        if (batches.size() == 1) {
            this->get_torch_tensor_from_eigen_matrix(data, batches[0]);
            if (this->m_use_colors) {
                batches[0] = torch::upsample_bilinear2d(batches[0].view({-1, 3, TParams::image_height, TParams::image_width}).to(this->m_device), {32, 32}, false);
            } else {
                batches[0] = torch::upsample_bilinear2d(batches[0].view({-1, 1, TParams::image_height, TParams::image_width}).to(this->m_device), {32, 32}, false);
            }
        } else {
            for (size_t ind = 0; ind < batches.size(); ind++) {
                this->get_torch_tensor_from_eigen_matrix(data.middleRows(ind * TParentLoader::Options::batch_size, TParentLoader::Options::batch_size),
                                                         batches[ind]);
                if (this->m_use_colors) {
                    batches[ind] = torch::upsample_bilinear2d(batches[ind].view({-1, 3, TParams::image_height, TParams::image_width}).to(this->m_device), {32, 32}, false);
                } else {
                    batches[ind] = torch::upsample_bilinear2d(batches[ind].view({-1, 1, TParams::image_height, TParams::image_width}).to(this->m_device), {32, 32}, false);
                }
            }
        }

    }



    void eval(const MatrixXf_rm &data,
              MatrixXf_rm &descriptors,
              MatrixXf_rm &recon_loss,
              MatrixXf_rm &reconstructed_data) {
        AutoEncoder auto_encoder = std::static_pointer_cast<AutoEncoderImpl>(this->m_auto_encoder_module.ptr());

        torch::Tensor eval_data;
        this->get_torch_tensor_from_eigen_matrix(data, eval_data);

        std::vector<torch::Tensor> outputs;

        // Get the names below with the inspect_graph.py script applied on the generated graph_text.pb file.
        if (this->m_use_colors) {
            eval_data = torch::upsample_bilinear2d(torch::reshape(eval_data, {-1, 3, TParams::image_height, TParams::image_width}).to(this->m_device), {32, 32}, false);
        } else {
            eval_data = torch::upsample_bilinear2d(torch::reshape(eval_data, {-1, 1, TParams::image_height, TParams::image_width}).to(this->m_device), {32, 32}, false);
        }

        torch::Tensor descriptors_tensor;
        torch::Tensor reconstruction_tensor = auto_encoder->forward_get_latent(eval_data, descriptors_tensor);
        torch::Tensor loss_tensor{torch::norm(reconstruction_tensor - eval_data, 2, {1, 2, 3})};

        //std::cout << "eval (reconstruction tensor sizes) - " << reconstruction_tensor.sizes() << std::endl;
        // TODO put those lines in another function
        this->get_eigen_matrix_from_torch_tensor(descriptors_tensor.cpu(), descriptors);
        this->get_eigen_matrix_from_torch_tensor(loss_tensor.cpu(), recon_loss);
        // TODO To avoid next line if not needed
        this->get_eigen_matrix_from_torch_tensor(torch::upsample_bilinear2d(reconstruction_tensor.cpu(), {TParams::image_height , TParams::image_width}, false), reconstructed_data);
    }

protected:
    const bool m_use_colors;
};

#endif
