
# coding: utf-8

# # GAN

# In[3]:

import random
import numpy as np

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ### Real (R) world data
# Sampled from gaussian distribution

# In[22]:

def get_distribution_sampler(mu, sigma):
    return lambda n: torch.Tensor(np.random.normal(mu, sigma, (1, n)))


# ### Input (I) to generator
# Sampled from uniform distribution making it much difficult for G to simply shift and scale R.

# In[28]:

def get_generator_input_sampler():
    return lambda n, m: torch.Tensor(n, m)


# ### Generator

# In[40]:

class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = F.elu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return x


# ### Discriminator

# In[46]:

class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x


# In[51]:

D = Discriminator(1, 10, 2)
D


# In[55]:

def decorate_with_diffs(data, exponent):
    mean = torch.mean(data.data, 1)
    mean_broadcast = torch.mul(torch.ones(data.size()), mean.tolist()[0][0])
    diffs = torch.pow(data - Variable(mean_broadcast), exponent)
    return torch.cat([data, diffs], 1)


# In[66]:

data = Variable(get_distribution_sampler(0, 1)(10))
data


# In[67]:

decorate_with_diffs(data, 1)


# In[ ]:



