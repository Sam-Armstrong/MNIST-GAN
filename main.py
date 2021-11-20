"""
Author: Sam Armstrong
Date: Autumn 2021

Description: The main body of code for training a generative adversarial network for generating handwritten digits (trained using MNIST)
"""

import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.autograd import Variable
from torch import Tensor
import time
from Discriminator import Discriminator
from Generator import Generator
import matplotlib.pyplot as plt

def wasserstein_loss(y_true, y_pred):
    return torch.mean(y_true * y_pred)

def boundary_seeking_loss(y_pred, y_true):
    """
    Boundary seeking loss.
    Reference: https://wiseodd.github.io/techblog/2017/03/07/boundary-seeking-gan/
    """
    return 0.5 * torch.mean((torch.log(y_pred) - torch.log(1 - y_pred)) ** 2)

def train():
    batch_size = 10
    num_epochs = 8

    plot_data = np.empty((num_epochs, 2), dtype = float)
    
    start_time = time.time()
    device = torch.device('cuda')

    # Loads the train and test data into PyTorch tensors
    training_data = datasets.MNIST(root = "data", train = True, download = True, transform = ToTensor())
    test_data = datasets.MNIST(root = "data", train = False, download = True, transform = ToTensor())

    # Loads the data into batches 
    train_dataloader = DataLoader(training_data, batch_size = batch_size, shuffle = True)

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    
    d_loss = nn.BCELoss() # Discriminator loss function

    # Optimizers
    #generator_optimizer = optim.SGD(generator.parameters(), lr = 3e-5)
    generator_optimizer = optim.RMSprop(generator.parameters(), lr = 1e-6, momentum = 0, weight_decay = 1e-4)
    discriminator_optimizer = optim.RMSprop(discriminator.parameters(), lr = 5e-7, momentum = 0, weight_decay = 1e-4) #optim.SGD(discriminator.parameters(), lr = 3e-5)

    
    for epoch in range(num_epochs):
        print('Epoch: ', epoch + 1)
        total_d_loss = 0
        total_g_loss = 0

        for i, (imgs, _) in enumerate(train_dataloader):

            # Adversarial ground truths
            valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0).to(device), requires_grad = False)
            fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0).to(device), requires_grad = False)
            real_imgs = imgs.type(Tensor).to(device) ##
            
            gen_input = Tensor(np.random.rand(batch_size, 16)).to(device) # Noise for the generator input ##
            gen_imgs = generator(gen_input) # Generate a batch of images

            generator_optimizer.zero_grad()
            g_loss = 1 - torch.mean(discriminator(gen_imgs))# * fake)
            g_loss.backward()
            generator_optimizer.step()
            total_g_loss += g_loss.item()

            discriminator_optimizer.zero_grad()
            real_loss = 1 - torch.mean(discriminator(real_imgs))# * valid)
            fake_loss = torch.mean(discriminator(gen_imgs.detach()))# * fake)
            dis_loss = (real_loss + fake_loss) / 2
            dis_loss.backward()
            discriminator_optimizer.step()
            total_d_loss += dis_loss.item()

            # ## Training the Generator
            # generator_optimizer.zero_grad()
            # gen_input = Variable(Tensor(np.random.rand(batch_size, 16)).to(device)) # Noise for the generator input
            # gen_imgs = generator(gen_input) # Generate a batch of images
            # g_loss = wasserstein_loss(discriminator(gen_imgs), valid) #boundary_seeking_loss(discriminator(gen_imgs), valid) # Loss measures the generator's ability to fool the discriminator
            # g_loss.backward()
            # generator_optimizer.step()
            # total_g_loss += g_loss.item()

            # ## Training the Discriminator
            # # Measure discriminator's ability to classify real from generated samples
            # discriminator_optimizer.zero_grad()
            # real_loss = d_loss(discriminator(real_imgs), valid)
            # fake_loss = d_loss(discriminator(gen_imgs.detach()), fake)
            # dis_loss = (real_loss + fake_loss) / 2
            # dis_loss.backward()
            # discriminator_optimizer.step()
            # total_d_loss += dis_loss.item()
        
        print('Generator Loss: ', total_g_loss / i)
        print('Discriminator Loss: ', total_d_loss / i)
        plot_data[epoch][0] = total_g_loss / i
        plot_data[epoch][1] = total_d_loss / i

    
    torch.save(generator.state_dict(), 'generator-model.pickle')
    print('Generator saved to .pickle file')
    print('Finsished in %s seconds' % str(round(time.time() - start_time, 2)))

    plt.plot(plot_data)
    plt.ylabel('Discriminator Loss')
    plt.xlabel('Epoch')
    plt.show()


if __name__ == "__main__":
    train()
