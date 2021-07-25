# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


class discriminator(nn.Module):
    def __init__(self, input_dim, num_classes):
        self.input_dim = input_dim
        self.num_classes = num_classes
        super(discriminator, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(input_dim, input_dim//2),
            nn.ReLU(),
            nn.Linear(input_dim//2, input_dim//4),
            nn.ReLU(),
            nn.Linear(input_dim//4, num_classes),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.layer(x)
        print(x.shape)
        return torch.clamp(x, 0., 1.)


class generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(generator, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.ReLU(),
            nn.Linear(input_dim * 2, input_dim * 4),
            nn.ReLU(),
            nn.Linear(input_dim * 4, output_dim),
        )

    def forward(self, x):
        x = self.layer(x)
        print(x.shape)
        return torch.clamp(x, 0., 1.)

