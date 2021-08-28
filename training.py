import pandas as pd
import numpy as np
import glob
import pickle
import matplotlib.pyplot as plt
import math
from scipy.stats import mode
import time
import gzip
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, TensorDataset

device = torch.device("cuda")

''' Things to try
1. Larger Network
2. Better normalization
3. Learning rate
4. Change the architecture to use a CNN
'''

'''
Things to try FOR REAL
1. Batchnorm
2. Adjusting hyper parameters
3. Doubling the outputs and inputs in the generator
'''

print("This is a 3-layer linear NN, adjusted, and 100 epochs.")

start_time = time.perf_counter()

#define your models
class Discriminator(nn.Module):
    def __init__(self, image_dimension):
        super().__init__()
        self.disc = nn.Sequential(
            #nn.Conv2d(image_dimension, 128, kernel_size = 3, stride = 2, padding = 1),
            #nn.BatchNorm2d(128),
            #nn.LeakyReLU(0.1, inplace=True),
            #nn.Conv2d(128, 256, kernel_size = 3, stride = 2, padding = 1),
            #nn.BatchNorm2d(256),
            #nn.LeakyReLU(0.1, inplace=True),
            #nn.Conv2d(256, 512, kernel_size = 3, stride = 2, padding = 1),
            #nn.BatchNorm2d(512),
            #nn.LeakyReLU(0.1, inplace=True),
            #nn.Conv2d(512, 1, kernel_size = 3, stride = 2, padding = 1),
            #nn.Sigmoid())
            nn.Linear(image_dimension, 128), 
            #nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Linear(128,64),
            #nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Linear(64,1),
            #nn.LeakyReLU(0.1),  
            #nn.Linear(10,1), 
            nn.Sigmoid())
        
    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, noise_dimension, image_dimension):
        super().__init__()
        self.gen = nn.Sequential(
            #nn.Conv2d(noise_dimension, 1024, kernel_size = 3, stride = 1, padding = 1),
            #nn.BatchNorm2d(1024),
            #nn.ReLU(True),
            #nn.Conv2d(1024, 512, kernel_size = 3, stride = 2, padding = 1),
            #nn.BatchNorm2d(512),
            #nn.ReLU(True),
            #nn.Conv2d(512, 256, kernel_size = 3, stride = 2, padding = 1),
            #nn.BatchNorm2d(256),
            #nn.ReLU(True),
            #nn.Conv2d(256, image_dimension, kernel_size = 3, stride = 2, padding = 1),
            #nn.Tanh())
            nn.Linear(noise_dimension, 64), 
            #nn.BatchNorm1d(64),
            #nn.Linear(10,64),
            #nn.LeakyReLU(0.1), 
            nn.LeakyReLU(0.1), 
            nn.Linear(64,128),
            #nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1), 
            nn.Linear(128,image_dimension),
            nn.Tanh())

    def forward(self, x):
        return self.gen(x)

# Hyper parameters
learning_rate = 0.0002
noise_dimension = 256
image_dimension = 28 * 28 * 1
batch_size = 25
num_epochs = 100

discriminator_model = Discriminator(image_dimension).to(device)
generator_model = Generator(noise_dimension, image_dimension).to(device)
fixed_noise = torch.randn((batch_size, noise_dimension)).to(device)
transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)
trainset = datasets.MNIST(root="dataset/", transform=transforms, download=True)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

optimizer_discriminator = optim.Adam(discriminator_model.parameters(), lr=learning_rate)
optimizer_generator = optim.Adam(generator_model.parameters(), lr=learning_rate)
criterion = nn.BCELoss()


def test_generator(e, model):
    model_name = "linear_model_3_layers_adj_" + str(e) + "_epochs.pth"
    torch.save(model.state_dict(),model_name)

    # Generate images
    np.random.seed(504)
    h = w = 28
    num_gen = 25

    z = np.random.normal(size=[num_gen, noise_dimension])

    #with torch.no_grad():
    noise = torch.randn((batch_size, noise_dimension)).to(device)
    generated_images = model(noise).reshape(-1,1,28,28)
    generated_images = generated_images.cpu().detach().numpy()

    # plot of generation
    n = np.sqrt(num_gen).astype(np.int32)
    I_generated = np.empty((h*n, w*n))
    for i in range(n):
        for j in range(n):
            I_generated[i*h:(i+1)*h, j*w:(j+1)*w] = generated_images[i*n+j, :].reshape(28, 28)

    plt.figure(figsize=(4, 4))
    plt.axis("off")
    plt.imshow(I_generated, cmap='gray')
    figure_name = "linear_3_layers_images_adj_" + str(e) + "_epochs.png"
    plt.savefig(figure_name)
    plt.show()
    plt.close()

# Training
def train():
    disc_losses = []
    generator_losses = []
    epochs = []

    for epoch in range(num_epochs):  
	
        # Create a batch by drawing random index numbers from the training set
        epochs.append(epoch+1)
        disc_loss = -1
        generator_loss = -1
        for batch_index, (real, _) in enumerate(trainloader):
            real = real.view(-1, image_dimension).to(device)
            #real = real.to(device)
            # Create noise vectors for the generator
            noise = torch.randn(batch_size, noise_dimension).to(device)

            # Generate the images from the noise
            fake = generator_model(noise)       

            # Create labels
            real_discriminate = discriminator_model(real).reshape(-1)
            disc_real_loss = criterion(real_discriminate, torch.ones_like(real_discriminate))
            fake_discriminate = discriminator_model(fake).reshape(-1)
            disc_fake_loss = criterion(fake_discriminate, torch.zeros_like(fake_discriminate))

            # Train discriminator on generated images
            disc_loss = (disc_real_loss + disc_fake_loss) / 2
            discriminator_model.zero_grad()
            disc_loss.backward(retain_graph=True)
            optimizer_discriminator.step()

            # Re-classify the images based on the updated discriminator
            labels = discriminator_model(fake).reshape(-1)

            # Train generator on discriminator results
            generator_loss = criterion(fake_discriminate, torch.ones_like(fake_discriminate))
            generator_model.zero_grad()
            generator_loss.backward()
            optimizer_generator.step()

            if batch_index == 0:
                print('[%d] Discriminator loss: %.3f, Generator Loss: %.3f' %
                    (epoch + 1, disc_loss, generator_loss))
        
        disc_losses.append(disc_loss)
        generator_losses.append(generator_loss)

        with open("losses.pickle", 'wb') as f:
            pickle.dump([epochs, disc_losses, generator_losses], f)

        plt.plot(epochs, disc_losses, color="red", linewidth = 2)
        plt.plot(epochs, generator_losses, color="blue", linewidth = 2)

        plt.xlabel('Epochs')
        plt.ylabel('Training Loss') 
        plt.legend(["Discriminator Loss", "Generator Loss"], loc = "upper right")
        plt.title('Discriminator and Generator Losses During Training')

        if epoch % 5 == 0:
            filename = "training_loss_plot_linear_3_layers_adj_" + str(epoch+1) + "_epochs.png"
            plt.savefig(filename)
            plt.show()
            plt.close()
            test_generator((epoch+1), generator_model)


train()

print("Finished!")

end_time = time.perf_counter()
print(f"Training took {end_time - start_time:0.4f} seconds")
minutes = (end_time - start_time) / 60.0
print(f"Training took {minutes:0.4f} minutes")