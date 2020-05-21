//
// Created by Luca Grillotti
//

#ifndef EXAMPLE_PYTORCH_AUTOENCODER_HPP
#define EXAMPLE_PYTORCH_AUTOENCODER_HPP

#include <torch/torch.h>

#include "encoder.hpp"
#include "decoder.hpp"

struct AutoEncoderImpl : torch::nn::Module {
    AutoEncoderImpl(int image_width, int image_height, int latent_dim, bool use_colors = false) :
            m_encoder(Encoder(image_width, image_height, latent_dim, use_colors)),
            m_decoder(Decoder(image_width, image_height, latent_dim, use_colors)) {
        register_module("encoder", m_encoder);
        register_module("decoder", m_decoder);
    }

    torch::Tensor forward(const torch::Tensor &x) {
        return m_decoder(m_encoder(x));
    }

    torch::Tensor forward_get_latent(const torch::Tensor &input, torch::Tensor &corresponding_latent) {
        corresponding_latent = m_encoder(input);
        return m_decoder(corresponding_latent);
    }

    Encoder m_encoder;
    Decoder m_decoder;
};

TORCH_MODULE(AutoEncoder);

#endif //EXAMPLE_PYTORCH_AUTOENCODER_HPP
