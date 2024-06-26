"""
CNN-LSTM-based autoencoder learn reconstructionly. This
version fix the CNN kernel length to 3 and stride to 2.
The encoder use tradition 2D CNN on each frame, then
pointwise CNN reduce person dim to 1. Finally, LSTM
process the data along time.
The decoder repeat the embedding data to fit temporal
length, then pass through LSTM and a fully connect
layer. Converse to the encoder, it then input pointwise
CNN increase the person dim then 2D CNN on each frame.
"""

import torch
from .base import BaseVAE
from torch import nn
from torch.nn import functional as F
from typing import List


class VanillaVAE(BaseVAE):
    def __init__(self,
                 mice_num,
                 bodypart,
                 frame_len,
                 cnn_hidden,
                 lstm_hidden,
                 latent_dim,
                 **kwargs):
        super().__init__()
        self.cnn_hidden = cnn_hidden
        self.lstm_hidden = lstm_hidden

        self.coord = 2
        self.joints = len(bodypart)
        second_cnn_input_dim = 2 if len(bodypart)==3 else 6
        self.frame_len = frame_len
        self.mice_num = mice_num

        # encoder
        en_cnn_module = []
        in_dim = self.coord
        for h in self.cnn_hidden:
            en_cnn_module.append(
                torch.nn.Sequential(
                    torch.nn.Conv2d(in_dim,
                                    h,
                                   (3, 1),
                                   (2, 1)),
                    torch.nn.LeakyReLU(),
                )
            )
            in_dim = h
        
        self.cnn1 = torch.nn.Sequential(*en_cnn_module)

        # pointwise layer reduce person dim to 1
        # 2 for 2 person
        self.cnn2 = torch.nn.Conv2d(second_cnn_input_dim, 1, 1)
        self.lstm1 = torch.nn.LSTM(self.cnn_hidden[-1],
                                   self.lstm_hidden,
                                   num_layers=2,
                                   batch_first=True)
        
        self.fc_mu = nn.Linear(self.lstm_hidden, latent_dim)
        self.fc_var = nn.Linear(self.lstm_hidden, latent_dim)

        # decoder
        self.fc = torch.nn.Linear(latent_dim,
                                  self.lstm_hidden)
        self.lstm2 = torch.nn.LSTM(self.lstm_hidden,
                                   2*self.lstm_hidden,
                                   num_layers=2,
                                   batch_first=True)
        self.fc2 = torch.nn.Linear(2*self.lstm_hidden,
                                  self.cnn_hidden[-1])
        self.cnn3 = torch.nn.Conv2d(1, second_cnn_input_dim, 1)

        de_cnn_module = []
        self.cnn_hidden.reverse()
        for h in range(len(self.cnn_hidden)-1):
            de_cnn_module.append(
                torch.nn.Sequential(
                    torch.nn.ConvTranspose2d(self.cnn_hidden[h],
                                             self.cnn_hidden[h+1],
                                             (1, 3),
                                             (1, 2)),
                    torch.nn.BatchNorm2d(self.cnn_hidden[h+1]),
                    torch.nn.LeakyReLU(),
                )
            )
            
        de_cnn_module.append(
            torch.nn.Sequential(
                    torch.nn.ConvTranspose2d(self.cnn_hidden[-1],
                                             self.coord,
                                             (1, 3),
                                             (1, 2),
                                             output_padding=(0,1)),
                    torch.nn.LeakyReLU(),
                )
        )
        
        self.cnn4 = torch.nn.Sequential(*de_cnn_module)

    def forward(self, input: torch.tensor, **kwargs) -> List[torch.tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, mu, log_var, z]
    
    def encode(self, inputs: torch.tensor) -> List[torch.tensor]:
        """
        Encodes tahe input by passing through the encoder network
        and returns the embedding codes.
        inputs: (Tensor) Input tensor to encoder [B x C x J x F]
        """
        x = self.cnn1(inputs) # -> B x H_C x 2 x F
        x = self.cnn2(x.transpose(1,2)) # -> B x 1 x H_C x F
        x, (hs, cs) = self.lstm1(x.squeeze(1).transpose(1,2)) # hs: B x H_R

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(hs[-1])
        log_var = self.fc_var(hs[-1])

        return [mu, log_var]

    def decode(self, embedding: torch.tensor) -> torch.tensor:
        """
        Decode the given embedding codes back to input space.
        embedding: (Tensor) embedded codes from all layer [NumofLatent x B x H]
        """
        embedding = self.fc(embedding)
        x, (hs, cs) = self.lstm2(embedding.unsqueeze(1).repeat(1, self.frame_len, 1))
        x = x.reshape((-1, self.frame_len, 2*self.lstm_hidden))
        x = self.fc2(x) # -> B x F x H
        x = self.cnn3(x.unsqueeze(1)) # -> B x 2 x F x H
        x = self.cnn4(x.transpose(1,3)) # -> B x C x F x J

        return x.transpose(2,3)

    def reparameterize(self, mu: torch.tensor, logvar: torch.tensor) -> torch.tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu


    def loss_function(self, *args, **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        recons_loss = F.mse_loss(recons, input)

        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0
        )

        loss = recons_loss + kld_loss
        return {
            "loss": loss,
            "reconstruction_loss": recons_loss.detach(),
            "KLD": -kld_loss.detach(),
        }

    def predict_loss(self, *args, **kwargs):
        recon = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        
        recons_loss = F.mse_loss(recon, input, reduction="none")
        recons_loss = recons_loss.mean([1,2,3])

        kld_loss = -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1)

        loss = recons_loss + kld_loss
        return {
            "loss": loss.cpu(),
            "reconstruction_loss": recons_loss.cpu(),
            "KLD": -kld_loss.cpu(),
        }
