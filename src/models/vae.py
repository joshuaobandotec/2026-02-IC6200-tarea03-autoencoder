"""Variational autoencoder convolucional."""
import torch
import torch.nn as nn


def down(c_in, c_out):
    return nn.Sequential(nn.Conv2d(c_in, c_out, 4, 2, 1), nn.BatchNorm2d(c_out),
                         nn.LeakyReLU(0.2, inplace=True))


def up(c_in, c_out):
    return nn.Sequential(nn.ConvTranspose2d(c_in, c_out, 4, 2, 1), nn.BatchNorm2d(c_out),
                         nn.LeakyReLU(0.2, inplace=True))


class VAE(nn.Module):
    def __init__(self, in_channels=3, latent_dim=128, channels=(32, 64, 128, 256),
                 image_size=128):
        super().__init__()
        channels = list(channels)
        self.feat = image_size // (2 ** len(channels))
        self.last_c = channels[-1]
        flat = self.last_c * self.feat * self.feat

        enc, c = [], in_channels
        for ch in channels:
            enc.append(down(c, ch))
            c = ch
        self.encoder = nn.Sequential(*enc)
        self.fc_mu = nn.Linear(flat, latent_dim)
        self.fc_logvar = nn.Linear(flat, latent_dim)

        self.fc_dec = nn.Linear(latent_dim, flat)
        dec, rev = [], channels[::-1]
        for c_in, c_out in zip(rev[:-1], rev[1:]):
            dec.append(up(c_in, c_out))
        self.decoder = nn.Sequential(*dec)
        self.out = nn.Sequential(nn.ConvTranspose2d(rev[-1], in_channels, 4, 2, 1),
                                 nn.Sigmoid())

    def encode(self, x):
        h = self.encoder(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        if self.training:
            return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return mu  # en evaluación usamos la media: reconstrucción determinista

    def decode(self, z):
        h = self.fc_dec(z).view(-1, self.last_c, self.feat, self.feat)
        return self.out(self.decoder(h))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return {"x_hat": self.decode(z), "z": mu, "mu": mu, "logvar": logvar}
