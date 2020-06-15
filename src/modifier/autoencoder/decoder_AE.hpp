//
// Created by Luca Grillotti
//

#ifndef AE_DECODER_HPP
#define AE_DECODER_HPP

#include <torch/torch.h>

struct DecoderImpl : torch::nn::Module {
    DecoderImpl(int latent_dim, int de_hid_dim1, int de_hid_dim2, int output_dim) :
        m_linear_1(torch::nn::Linear(latent_dim, de_hid_dim1)),
        m_linear_2(torch::nn::Linear(de_hid_dim1, de_hid_dim2)),
        m_linear_3(torch::nn::Linear(de_hid_dim2, output_dim)),
        m_device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU)
        {
            register_module("linear_1", m_linear_1);
            register_module("linear_2", m_linear_2);
            register_module("linear_3", m_linear_3);
        }

        torch::Tensor forward(const torch::Tensor &z, torch::Tensor &tmp) 
        {
            return m_linear_3(torch::relu(m_linear_2(torch::relu(m_linear_1(z)))));
        }

        torch::nn::Linear m_linear_1, m_linear_2, m_linear_3;
        torch::Device m_device;
};

TORCH_MODULE(Decoder);

#endif //EXAMPLE_PYTORCH_DECODER_HPP