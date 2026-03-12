import torch
import torch.nn as nn


def down_block(in_ch, out_ch, bn=True):
    layers = [nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False)]
    if bn:
        layers.append(nn.BatchNorm2d(out_ch))
    layers.append(nn.LeakyReLU(0.2))
    return nn.Sequential(*layers)


def up_block(in_ch, out_ch, dropout=False):
    layers = [
        nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False),
        nn.BatchNorm2d(out_ch),
    ]
    if dropout:
        layers.append(nn.Dropout(0.5))
    layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = down_block(3, 64, bn=False)
        self.enc2 = down_block(64, 128)
        self.enc3 = down_block(128, 256)
        self.enc4 = down_block(256, 512)

        self.dec1 = up_block(512, 256)
        self.dec2 = up_block(512, 128)
        self.dec3 = up_block(256, 64)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, 3, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        d1 = self.dec1(e4)
        d2 = self.dec2(torch.cat([d1, e3], 1))
        d3 = self.dec3(torch.cat([d2, e2], 1))
        return self.final(torch.cat([d3, e1], 1))


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            down_block(6, 64, bn=False),
            down_block(64, 128),
            down_block(128, 256),
            nn.Conv2d(256, 1, 4, 1, 1),
        )

    def forward(self, inp, tar):
        return self.net(torch.cat([inp, tar], 1))