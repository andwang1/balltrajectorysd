import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class VAE_FC(nn.Module):
    def __init__(self, input_dim, en_hid_dim1, latent_dim, de_hid_dim1, de_hid_dim2, output_dim, beta, VERBOSE, device):
        super(VAE_FC, self).__init__()
        self.en_fc1 = nn.Linear(input_dim, en_hid_dim1)
        self.en_fc_m = nn.Linear(en_hid_dim1, latent_dim)
        self.en_fc_v = nn.Linear(en_hid_dim1, latent_dim)

        self.de_fc1 = nn.Linear(latent_dim, de_hid_dim1)
        self.de_fc2 = nn.Linear(de_hid_dim1, de_hid_dim2)
        self.de_fc_m = nn.Linear(de_hid_dim2, output_dim)
        self.de_fc_v = nn.Linear(de_hid_dim2, output_dim)

        self.beta = beta

        self.VERBOSE = VERBOSE
        self.device = device

    def encode(self, x):
        x_hid = F.relu(self.en_fc1(x))
        # Generate mu and logvar
        mu = self.en_fc_m(x_hid)
        logvar = self.en_fc_v(x_hid)
        return mu, logvar

    def reparametrize(self, mu, logvar):
        z = torch.randn(logvar.shape, requires_grad=True, device=self.device) * torch.exp(0.5 * logvar) + mu
        return z

    def decode(self, z):
        out = F.relu(self.de_fc2(F.relu(self.de_fc1(z))))
        out_mu = self.de_fc_m(out)
        out_logvar = self.de_fc_v(out)
        return out_mu, out_logvar

    def sample_output(self, mu, logvar):
        x = torch.randn(logvar.shape, requires_grad=True, device=self.device) * torch.exp(0.5 * logvar) + mu
        return x

    def forward(self, x):
        en_mu, en_logvar = self.encode(x)
        z = self.reparametrize(en_mu, en_logvar)
        de_mu, de_logvar = self.decode(z)
        return self.sample_output(de_mu, de_logvar), torch.exp(de_logvar), en_mu, en_logvar

    def loss_function(self, en_mu, en_logvar, de_logvar, pred_traj, label_traj, full_loss = True):
        KL_loss = -0.5 * torch.sum(1 + en_logvar - en_mu.pow(2) - torch.exp(en_logvar))
        L2_loss = F.mse_loss(pred_traj, label_traj, reduction="sum")
        recon_loss = torch.sum(((label_traj - pred_traj) ** 2) / (2 * (torch.exp(de_logvar) + 0.0001)) + 0.5 * (
                    de_logvar + np.log(2 * np.pi)))
        if self.VERBOSE:
            print("PRED", pred_traj)
            print("LABEL", label_traj)
            print("LOGVAR", de_logvar)
            print("VAR", torch.exp(de_logvar))
        if full_loss:
            loss = recon_loss + self.beta * KL_loss
        else:
            loss = L2_loss + self.beta * KL_loss
        return loss, KL_loss, L2_loss