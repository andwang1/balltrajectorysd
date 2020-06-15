//
// Created by Andy Wang
//

#ifndef AE_HPP
#define AE_HPP

#include <torch/torch.h>

#include "encoder_AE.hpp"
#include "decoder_AE.hpp"

struct AutoEncoderImpl : torch::nn::Module {
    AutoEncoderImpl(int input_dim, int en_hid_dim1, int en_hid_dim2, int latent_dim, int de_hid_dim1, int de_hid_dim2, int output_dim) :
            #ifdef AURORA
            m_encoder(Encoder(output_dim, en_hid_dim1, en_hid_dim2, latent_dim)),
            #else
            m_encoder(Encoder(input_dim, en_hid_dim1, en_hid_dim2, latent_dim)),
            #endif
            m_decoder(Decoder(latent_dim, de_hid_dim1, de_hid_dim2, output_dim)) 
    {
        register_module("encoder", m_encoder);
        register_module("decoder", m_decoder);
    }

    torch::Tensor forward(const torch::Tensor &x) {
        torch::Tensor encoder_mu, encoder_logvar, decoder_logvar;
        return m_decoder(m_encoder(x, encoder_mu, encoder_logvar), decoder_logvar);
    }

    torch::Tensor forward_get_latent(const torch::Tensor &input, torch::Tensor &encoder_mu, torch::Tensor &encoder_logvar, torch::Tensor &decoder_logvar, torch::Tensor &corresponding_latent) {
        corresponding_latent = m_encoder(input, encoder_mu, encoder_logvar);
        return m_decoder(corresponding_latent, decoder_logvar);
    }

    torch::Tensor forward_(const torch::Tensor &input, torch::Tensor &encoder_mu, torch::Tensor &encoder_logvar, torch::Tensor &decoder_logvar) {
        torch::Tensor corresponding_latent = m_encoder(input, encoder_mu, encoder_logvar);
        return m_decoder(corresponding_latent, decoder_logvar);
    }

    Encoder m_encoder;
    Decoder m_decoder;
};

TORCH_MODULE(AutoEncoder);

#endif //EXAMPLE_PYTORCH_AUTOENCODER_HPP
