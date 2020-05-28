//
// Created by Luca Grillotti
//

#ifndef EXAMPLE_PYTORCH_AUTOENCODER_HPP
#define EXAMPLE_PYTORCH_AUTOENCODER_HPP

#include <torch/torch.h>

#include "encoder.hpp"
#include "decoder.hpp"

struct AutoEncoderImpl : torch::nn::Module {
    AutoEncoderImpl(int input_dim, int en_hid_dim1, int latent_dim, int de_hid_dim1, int de_hid_dim2, int output_dim) :
            m_encoder(Encoder(input_dim, en_hid_dim1, latent_dim)),
            m_decoder(Decoder(latent_dim, de_hid_dim1, de_hid_dim2, output_dim)) {
        register_module("encoder", m_encoder);
        register_module("decoder", m_decoder);
    }

    torch::Tensor forward(const torch::Tensor &x) {
        torch::Tensor encoder_mu, encoder_logvar;
        // std::cout << x << "AE" << std::endl;
        return m_decoder(m_encoder(x, encoder_mu, encoder_logvar));
    }

    torch::Tensor forward_get_latent(const torch::Tensor &input, torch::Tensor &corresponding_latent) {
        torch::Tensor encoder_mu, encoder_logvar;
        corresponding_latent = m_encoder(input, encoder_mu, encoder_logvar);
        return m_decoder(corresponding_latent);
    }

    torch::Tensor forward_get_encoder_stats(const torch::Tensor &input, torch::Tensor &encoder_mu, torch::Tensor &encoder_logvar) {
        torch::Tensor corresponding_latent = m_encoder(input, encoder_mu, encoder_logvar);
        return m_decoder(corresponding_latent);
    }

    Encoder m_encoder;
    Decoder m_decoder;
};

TORCH_MODULE(AutoEncoder);

#endif //EXAMPLE_PYTORCH_AUTOENCODER_HPP
