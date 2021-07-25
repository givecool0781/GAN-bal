# -*- coding: utf-8 -*-
import time
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms
from model import discriminator, generator
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch.utils.data as utils
from   preprocessing import load_data

start_time = time.time()



# Discriminator Loss => BCELoss
def d_loss_function(inputs, targets):
    return nn.BCELoss()(inputs, targets)


def g_loss_function(inputs):
    targets = torch.ones([inputs.shape[0], 1])
    targets = targets.to(device)
    return nn.BCELoss()(inputs, targets)

# GPU
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('GPU State:', device)

# Load data
X,Y= load_data()
input_dim = X.shape[1]
num_class = len(np.unique(Y))

# Model
G = generator(input_dim, input_dim).to(device)
D = discriminator(input_dim, num_class).to(device)
print(G)
print(D)

# Settings
epochs = 200
lr = 0.0002
batch_size = 1024
g_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
d_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
SEED = 42

# Transform
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])




sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=SEED)
for dev_index, val_index in sss.split(X, Y):  # runs only once
    X_dev = X[dev_index]
    Y_dev = Y[dev_index]
    X_val = X[val_index]
    Y_val = Y[val_index]

# Train Data
tensor_x = torch.stack([torch.Tensor(i) for i in X_dev]).to(device)
tensor_y = torch.LongTensor(Y_dev).to(device)  # checked working correctly
dataset = utils.TensorDataset(tensor_x, tensor_y)
train_loader = utils.DataLoader(dataset, batch_size=batch_size)
#Test Data
tensor_x_val = torch.stack([torch.Tensor(i) for i in X_val]).to(device)
tensor_y_val = torch.LongTensor(Y_val).to(device)
dataset_val = utils.TensorDataset(tensor_x_val,tensor_y_val)
test_loader = utils.DataLoader(dataset_val,batch_size=batch_size)
# print(train_loader)

# train_set = datasets.MNIST('mnist/', train=True, download=True, transform=transform)
# test_set = datasets.MNIST('mnist/', train=False, download=True, transform=transform)
# train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# Train
for epoch in range(epochs):
    epoch += 1

    for times, data in enumerate(train_loader):
        times += 1
        real_inputs = data[0].to(device)
        test = 255 * (0.5 * real_inputs[0] + 0.5)
        print(real_inputs.shape)
        # real_inputs = real_inputs.view(-1, 896)
        real_outputs = D(real_inputs)
        real_label = torch.ones(real_inputs.shape[0], 1).to(device)

        noise = (torch.rand(real_inputs.shape[0], 128) - 0.5) / 0.5
        noise = noise.to(device)
        fake_inputs = G(noise)
        fake_outputs = D(fake_inputs)
        fake_label = torch.zeros(fake_inputs.shape[0], 1).to(device)

        outputs = torch.cat((real_outputs, fake_outputs), 0)
        targets = torch.cat((real_label, fake_label), 0)

        # Zero the parameter gradients
        d_optimizer.zero_grad()

        # Backward propagation
        d_loss = d_loss_function(outputs, targets)
        d_loss.backward()
        d_optimizer.step()

        # Generator
        noise = (torch.rand(real_inputs.shape[0], 128)-0.5)/0.5
        noise = noise.to(device)

        fake_inputs = G(noise)
        fake_outputs = D(fake_inputs)

        g_loss = g_loss_function(fake_outputs)
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        if times % 100 == 0 or times == len(train_loader):
            print('[{}/{}, {}/{}] D_loss: {:.3f} G_loss: {:.3f}'.format(epoch, epochs, times, len(train_loader), d_loss.item(), g_loss.item()))

        print(fake_inputs)


    if epoch % 50 == 0:
        torch.save(G, 'Generator_epoch_{}.pth'.format(epoch))
        print('Model saved.')

    # imgs_numpy = (fake_inputs.data.cpu().numpy() + 1.0) / 2.0
    # show_images(imgs_numpy[:16])
    # plt.show()

print('Training Finished.')
print('Cost Time: {}s'.format(time.time()-start_time))