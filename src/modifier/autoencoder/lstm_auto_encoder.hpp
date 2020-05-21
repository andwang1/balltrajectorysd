//
// Created by Luca Grillotti on 20/05/2020.
//

#ifndef EXAMPLE_PYTORCH_LSTM_AUTO_ENCODER_HPP
#define EXAMPLE_PYTORCH_LSTM_AUTO_ENCODER_HPP

#include <torch/torch.h>

struct LSTMEncoderImpl : torch::nn::Module {

    LSTMEncoderImpl(size_t input_size, size_t latent_space_per_layer, size_t number_layers) {
        torch::nn::LSTMOptions lstm_options(input_size, latent_space_per_layer);
        lstm_options.batch_first(true);
        lstm_options.layers(number_layers);
        m_lstm = torch::nn::LSTM(lstm_options);
        register_module("LSTM", m_lstm);
    }

    torch::nn::RNNOutput forward(const torch::Tensor &x) {
        return m_lstm->forward(x);
    }

    torch::nn::LSTM m_lstm{nullptr};
};

TORCH_MODULE(LSTMEncoder);

struct LSTMDecoderImpl : torch::nn::Module {
    LSTMDecoderImpl(size_t input_size, size_t latent_space_per_layer, size_t number_layers) {
        torch::nn::LSTMOptions lstm_options(input_size, latent_space_per_layer);
        lstm_options.batch_first(true);
        lstm_options.layers(number_layers);
        m_lstm = torch::nn::LSTM(lstm_options);

        m_linear = torch::nn::Linear(latent_space_per_layer, input_size);

        register_module("LSTM", m_lstm);
        register_module("Linear", m_linear);
    }

    torch::Tensor forward(const torch::Tensor &x,
                          const torch::Tensor &hidden) {
        torch::Tensor out_first = m_linear->forward(hidden.select(0, 0).select(0,m_lstm->options.layers() - 1).unsqueeze(1));
        torch::nn::RNNOutput lstm_output = m_lstm->forward(x.slice(1, 1).flip(1), hidden);
        torch::Tensor out = m_linear->forward(lstm_output.output);
        return torch::cat({out_first, out}, 1).flip(1);
    }

    torch::nn::LSTM m_lstm{nullptr};
    torch::nn::Linear m_linear{nullptr};
};

TORCH_MODULE(LSTMDecoder);


struct LSTMAutoencoderImpl : torch::nn::Module {

    LSTMAutoencoderImpl(size_t input_size, size_t latent_space_per_layer, size_t number_layers)  {
        m_encoder = LSTMEncoder(input_size, latent_space_per_layer, number_layers);
        m_decoder = LSTMDecoder(input_size, latent_space_per_layer, number_layers);

        register_module("LSTMEncoder", m_encoder);
        register_module("LSTMDecoder", m_decoder);
    }

    torch::Tensor forward(const torch::Tensor &x)  {
        torch::nn::RNNOutput encoding = m_encoder->forward(x);
        return m_decoder->forward(x, encoding.state);
    }

    torch::Tensor forward_get_latent(const torch::Tensor &x, torch::Tensor &latent) {
        torch::nn::RNNOutput encoding = m_encoder->forward(x);
        latent = encoding.state.select(0, 1).flatten();
        return m_decoder->forward(x, encoding.state);
    }

    LSTMEncoder encoder() {
        return m_encoder;
    }

    LSTMDecoder decoder() {
        return m_decoder;
    }

    LSTMEncoder m_encoder{nullptr};
    LSTMDecoder m_decoder{nullptr};
};

TORCH_MODULE(LSTMAutoencoder);



#endif //EXAMPLE_PYTORCH_LSTM_AUTO_ENCODER_HPP
