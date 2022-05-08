#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os 
import time
import random
from packaging import version
import re
import pandas as pd

from tqdm import tqdm

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score


# In[2]:


input_neurons = 9
output_neurons = 4
hidden_neurons= 6
hidden_layer = 1


# In[3]:


# setting device as GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)


# In[4]:


def init_weights(layer):
    if type(layer) == nn.Linear:
        torch.nn.init.xavier_normal_(layer.weight)
        layer.bias.data.fill_(0.01)
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.build_architecture()
        self.apply(init_weights)

    def build_architecture(self):
        self.encode = nn.Sequential(
            nn.Linear(input_neurons, hidden_neurons),
            nn.Tanh(),
            nn.Linear(hidden_neurons, output_neurons)
        )
        self.decode = nn.Sequential(
            nn.Tanh(),
            nn.Linear( output_neurons, hidden_neurons),
            nn.Tanh(),
            nn.Linear(hidden_neurons, input_neurons),
        )
 
    def forward(self, batch: torch.tensor):
        return self.decode(self.encode(batch))


# In[5]:


learning_rate = 0.001
batch_size = 64
num_epochs = 20


# In[6]:


# Load Training and Test data
location = os.path.join('', './data/SpotifyFeatures.csv')
  
data = pd.read_csv(location)
data = data.drop_duplicates(subset=['track_id'])
data = data[data['track_id'] != id]

features_used = data[["genre","artist_name","track_name","track_id", "acousticness", "danceability","energy", "instrumentalness", "liveness", "loudness", "speechiness", "tempo", "valence"]]

original_data_all = np.array(features_used.values)

np.random.seed(1)
np.random.shuffle(original_data_all)

original_data1 = original_data_all[:,4:]
print(original_data1.shape)
original_data = np.array(original_data1,"float64")

train_data = original_data[:176771,:]
test_data = original_data[176771:,:]
train_torch = torch.from_numpy(train_data)
test_torch = torch.from_numpy(test_data)

#Dataloader
train_loader = DataLoader(dataset= train_torch, batch_size = batch_size)
test_loader = DataLoader(dataset= test_torch, batch_size = batch_size)


# In[7]:


activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


# In[8]:


#Initialize model
model = Autoencoder().to(device)
model.encode.register_forward_hook(get_activation('encode'))


# In[9]:


criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)


# In[10]:


#Train model
extracted = torch.DoubleTensor([[0,0,0,0]]).to(device)
for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device=device)
            labels = batch
            optimizer.zero_grad()
            outputs = model(batch.float())
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            if epoch == num_epochs-1:
                extracted = torch.cat((extracted, activation['encode']), 0)
        epoch_loss = epoch_loss / len(train_loader)
        print(f"Loss of epoch {epoch + 1}: {epoch_loss}")
print(extracted)


# In[11]:


extracted = extracted[1:,:] 


# In[12]:


extracted_test = torch.DoubleTensor([[0,0,0,0]]).to(device)
for batch in test_loader:
    batch = batch.to(device=device)
    labels = batch
    outputs = model(batch.float())
    extracted_test = torch.cat((extracted_test, activation['encode']), 0)


# In[13]:


extracted_test = extracted_test[1:,:] 


# In[14]:


extracted_test = torch.Tensor.cpu(extracted_test)
extracted_test = extracted_test.numpy()
extracted = torch.Tensor.cpu(extracted)
extracted = extracted.numpy()


# In[17]:


#knn 
k = 2
nearest_ids = []
for tens in extracted_test:
    distances = np.linalg.norm(extracted - tens, axis=1) # to compute euclidean distance
    nearest_neighbors_ids = distances.argsort()[:k] # to find nearest k neighbors,
    nearest_ids.append(nearest_neighbors_ids)


# In[24]:


for i in range(len(nearest_ids)):
    result = tuple(original_data_all[:,2][nearest_ids[i]][:k])
    print(result)


# In[ ]:




