//
// Created by Luca Grillotti
//

#ifndef VAE_ENCODER_HPP
#define VAE_ENCODER_HPP

struct EncoderImpl : torch::nn::Module {
    EncoderImpl(int input_dim, int en_hid_dim1, int en_hid_dim2, int latent_dim, bool bias) :
        m_linear_1(torch::nn::LinearOptions(input_dim, en_hid_dim1).bias(true)),
        m_linear_2(torch::nn::LinearOptions(en_hid_dim1, en_hid_dim2).bias(true)),
        m_linear_m(torch::nn::LinearOptions(en_hid_dim2, latent_dim).bias(true)),
        m_linear_v(torch::nn::LinearOptions(en_hid_dim2, latent_dim).bias(true)),
        m_device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU)
        {
            register_module("linear_1", m_linear_1);
            register_module("linear_2", m_linear_2);
            register_module("linear_m", m_linear_m);
            register_module("linear_v", m_linear_v);
        }

        void encode(const torch::Tensor &x, torch::Tensor &mu, torch::Tensor &logvar)
        {
            torch::Tensor out;
            out = torch::relu(m_linear_2(torch::relu(m_linear_1(x))));
            mu = m_linear_m(out);
            logvar = m_linear_v(out);
        }

        void reparametrize(const torch::Tensor &mu, const torch::Tensor &logvar, torch::Tensor &z)
        {
            z = torch::randn_like(logvar, torch::device(m_device).requires_grad(true)) * torch::exp(0.5 * logvar) + mu;
        }


        torch::Tensor forward(const torch::Tensor &x, torch::Tensor &encoder_mu, torch::Tensor &encoder_logvar, bool sample)
        {
            encode(x, encoder_mu, encoder_logvar);
            if (sample)
            {
                torch::Tensor z;
                reparametrize(encoder_mu, encoder_logvar, z);
                return z;
            }
            else
                {return encoder_mu;}
        }


        torch::nn::Linear m_linear_1, m_linear_2, m_linear_m, m_linear_v;
        torch::Device m_device;
};

TORCH_MODULE(Encoder);

#endif //EXAMPLE_PYTORCH_ENCODER_HPP
