#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

input_neurons = 9
output_neurons = 4
hidden_neurons = 6
hidden_layer = 1
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook
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

def NN(track, learning_rate= 0.001,batch_size = 64,num_epochs=20,return_full=False):
    
    # setting device as GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # Load Training and Test data
    dirname = os.path.dirname(__file__)
    location = os.path.join(dirname, '../data/SpotifyFeatures.csv')
    
    selected = np.array(
        [track['acousticness'], track['danceability'], track['energy'], track['instrumentalness'], track['liveness'],
         track['loudness'], track['speechiness'], track['tempo'], track['valence']], dtype='float64')
    
    id = track['id']

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
    train_data = original_data[:,:]
    test_data = np.array([selected])
    train_torch = torch.from_numpy(train_data)
    test_torch = torch.from_numpy(test_data)
    
    #Dataloader
    train_loader = DataLoader(dataset= train_torch, batch_size = batch_size)
    test_loader = DataLoader(dataset= test_torch, batch_size = batch_size)
    extracted = torch.DoubleTensor([[0,0,0,0]]).to(device)
    """
    #Initialize model
    model = Autoencoder().to(device)
    model.encode.register_forward_hook(get_activation('encode'))

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

    #Train model
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
    extracted = extracted[1:,:]
    extracted = torch.Tensor.cpu(extracted)
    extracted = extracted.numpy()
    np.savetxt("foo.csv", extracted, delimiter=",")
    torch.save(model.state_dict(), "./model.pth")
   """
    #load model
    model = Autoencoder().to(device)
    model.load_state_dict(torch.load("./model.pth"), strict= False)
    model.encode.register_forward_hook(get_activation('encode'))
    #read foo
    extracted = pd.read_csv('./foo.csv')
    extracted = np.array(extracted,"float64")

    extracted_test = torch.DoubleTensor([[0,0,0,0]]).to(device)
    for batch in test_loader:
        batch = batch.to(device=device)
        labels = batch
        outputs = model(batch.float())
        extracted_test = torch.cat((extracted_test, activation['encode']), 0)
    extracted_test = extracted_test[1:,:]

    extracted_test = torch.Tensor.cpu(extracted_test)
    extracted_test = extracted_test.numpy()

    #knn
    k = 10
    nearest_ids = []
    for tens in extracted_test:
        distances = np.linalg.norm(extracted - tens, axis=1) # to compute euclidean distance
        nearest_neighbors_ids = distances.argsort()[:k] # to find nearest k neighbors,
        nearest_ids.append(nearest_neighbors_ids)

    if return_full:
        result = tuple(original_data_all[:, 4:][nearest_neighbors_ids][:k])
    else:
        result = tuple(original_data_all[:, 3][nearest_neighbors_ids][:k])
    return result





