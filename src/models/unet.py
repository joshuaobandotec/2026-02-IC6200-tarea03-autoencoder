"""Autoencoder U-Net con skip connections."""
import torch
import torch.nn as nn


def conv_block(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, 1, 1), nn.BatchNorm2d(c_out), nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(c_out, c_out, 3, 1, 1), nn.BatchNorm2d(c_out), nn.LeakyReLU(0.2, inplace=True),
    )


class UNetAE(nn.Module):
    def __init__(self, in_channels=3, latent_dim=128, channels=(32, 64, 128, 256),
                 image_size=128, use_skips=(True, True, True, True)):
        super().__init__()
        channels = list(channels)
        self.use_skips = list(use_skips)
        assert len(self.use_skips) == len(channels), "use_skips debe tener un valor por nivel"
        self.feat = image_size // (2 ** len(channels))
        self.last_c = channels[-1]
        flat = self.last_c * self.feat * self.feat

        # Encoder: conv_block (guarda skip) + downsample
        self.enc_blocks, c = nn.ModuleList(), in_channels
        for ch in channels:
            self.enc_blocks.append(conv_block(c, ch))
            c = ch
        self.pool = nn.MaxPool2d(2)

        # Cuello de botella: vector latente z
        self.fc_enc = nn.Linear(flat, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, flat)

        # Decoder: upsample + concat(skip) + conv_block
        self.ups, self.dec_blocks = nn.ModuleList(), nn.ModuleList()
        c = channels[-1]
        for i, ch in reversed(list(enumerate(channels))):
            self.ups.append(nn.ConvTranspose2d(c, ch, 2, 2))
            c_in = ch * 2 if self.use_skips[i] else ch
            self.dec_blocks.append(conv_block(c_in, ch))
            c = ch
        self.out = nn.Sequential(nn.Conv2d(channels[0], in_channels, 1), nn.Sigmoid())

    def encode(self, x):
        skips, h = [], x
        for block in self.enc_blocks:
            h = block(h)
            skips.append(h)
            h = self.pool(h)
        z = self.fc_enc(h.flatten(1))
        return z, skips

    def forward(self, x):
        z, skips = self.encode(x)
        h = self.fc_dec(z).view(-1, self.last_c, self.feat, self.feat)
        n = len(skips)
        for j, (up, block) in enumerate(zip(self.ups, self.dec_blocks)):
            i = n - 1 - j  # nivel correspondiente del encoder
            h = up(h)
            if self.use_skips[i]:
                h = torch.cat([h, skips[i]], dim=1)
            h = block(h)
        return {"x_hat": self.out(h), "z": z}
