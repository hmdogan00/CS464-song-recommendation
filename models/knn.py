import numpy as np
import sys
import pandas as pd
import os
def knn(track, k=2, dist_metric='l2' ):
  dirname = os.path.dirname(__file__)
  location = os.path.join(dirname, '../data/SpotifyFeatures.csv')
  
  selected = np.array([track['acousticness'], track['danceability'], track['energy'], track['instrumentalness'], track['liveness'], track['loudness'], track['speechiness'], track['tempo'], track['valence']], dtype='float64')
  id = track['id']

  data = pd.read_csv(location)
  data = data.drop_duplicates(subset=['track_id'])
  data = data[data['track_id'] != id]

  features_used = data[["genre","artist_name","track_name","track_id", "acousticness", "danceability","energy", "instrumentalness", "liveness", "loudness", "speechiness", "tempo", "valence"]]

  original_data_all = np.array(features_used.values)

  np.random.seed(1)
  np.random.shuffle(original_data_all)

  original_data1 = original_data_all[:,4:]

  original_data = np.array(original_data1,"float64")

  train_data = original_data[:,:]
  if dist_metric == 'euclidian':
    distances = np.linalg.norm(train_data - selected, axis=1) # to compute euclidean distance
  else:
    distances = np.linalg.norm(train_data - selected, axis=1) # to compute euclidean distance

  nearest_neighbors_ids = distances.argsort()[:k] # to find nearest k neighbors
  result = tuple(original_data_all[:,3][nearest_neighbors_ids][:k])
  return result