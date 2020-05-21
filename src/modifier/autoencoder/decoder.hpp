//
// Created by Luca Grillotti
//

#ifndef EXAMPLE_PYTORCH_DECODER_HPP
#define EXAMPLE_PYTORCH_DECODER_HPP

#include <torch/torch.h>
/* TODO add padding to get a same padding */

struct DecoderImpl : torch::nn::Module {
    DecoderImpl(int image_width, int image_height, int latent_dim, bool use_colors=false) :
            m_linear_1(torch::nn::Linear(latent_dim, 32)),
            m_linear_2(torch::nn::Linear(32, 64)),
            m_deconv_1(torch::nn::Conv2d(
                    torch::nn::Conv2dOptions(4, 4, 3)
                            .padding(1)
                            .transposed(true))),
            m_deconv_2(torch::nn::Conv2d(
                    torch::nn::Conv2dOptions(4, 4, 3)
                            .padding(1)
                            .transposed(true)
            )),
            m_deconv_3(torch::nn::Conv2d(
                    torch::nn::Conv2dOptions(4, 4, 3)
                            .padding(1)
                            .transposed(true)
            )),
            m_deconv_4(torch::nn::Conv2d(
                    torch::nn::Conv2dOptions(4, use_colors ? 3 : 1, 3)
                            .padding(1)
                            .transposed(true)
            )) {
        register_module("linear_1", m_linear_1);
        register_module("linear_2", m_linear_2);

        register_module("deconv_1", m_deconv_1);
        register_module("deconv_2", m_deconv_2);
        register_module("deconv_3", m_deconv_3);
        register_module("deconv_4", m_deconv_4);
    }

    torch::Tensor forward(const torch::Tensor &x) {
        torch::Tensor output;
        output = torch::relu(m_linear_1(x));
        output = torch::relu(m_linear_2(output));
        output = torch::reshape(output, {-1, 4, 4, 4});
        output = torch::upsample_nearest2d(torch::relu(m_deconv_1(output)), {8, 8});
        output = torch::upsample_nearest2d(torch::relu(m_deconv_2(output)), {16, 16});
        output = torch::upsample_nearest2d(torch::relu(m_deconv_3(output)), {32, 32});
        output = m_deconv_4(output);
        return output;
    }

    torch::nn::Linear m_linear_1, m_linear_2;
    torch::nn::Conv2d m_deconv_1, m_deconv_2, m_deconv_3, m_deconv_4;
};

TORCH_MODULE(Decoder);

#endif //EXAMPLE_PYTORCH_DECODER_HPP
