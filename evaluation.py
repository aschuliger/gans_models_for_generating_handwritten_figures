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


def test_generator(noise, model, num):
    # Generate images
    np.random.seed(504)
    h = w = 28
    num_gen = 25
    batch_size = 25
    noise_dimension = 64

    #with torch.no_grad():
    #noise = torch.randn((batch_size, noise_dimension)).to(device)
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
    figure_name = "resulting_image.png"
    plt.savefig(figure_name)
    plt.show()
    plt.close()

path = sys.argv[1]
filenames = glob.glob(path + "/*.pickle")
cnt = 0

for filename in filenames:
    noise = pickle.load( open(filename, "rb") )
    cnt = cnt + 1
    model = Net().to(device)
    model.load_state_dict(torch.load("final_model.pth"))
    test_generator(noise, model, cnt)