import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
def pcaknn(track, k=10, n=3,return_full=False):
  dirname = os.path.dirname(__file__)
  location = os.path.join(dirname, '../data/SpotifyFeatures.csv')
  selected = np.array([track['acousticness'], track['danceability'], track['energy'], track['instrumentalness'], track['liveness'], track['loudness'], track['speechiness'], track['tempo'], track['valence']], dtype='float64')
  id = track['id']

  data = pd.read_csv(location)
  data = data.drop_duplicates(subset=['track_id'])
  data = data[data['track_id'] != id]

  features_all = data[['genre','artist_name',"track_name","track_id", "acousticness", "danceability","energy", "instrumentalness", "liveness", "loudness", "speechiness", "tempo", "valence"]]
  #apply PCA
  Y = np.array(features_all.values)
  X = Y[:,4:]
  #create PCA
  pca = PCA(n_components=n)
  princ_comps = pca.fit_transform(X)
  #new dataset by using principle components
  new_data = pd.DataFrame(data = princ_comps)
  original_data_all = np.array(new_data.values)
  original_data_all = np.c_[features_all["track_id"],original_data_all]

  original_data1 = original_data_all[:,1:]

  original_data = np.array(original_data1,"float64")


  train_data = original_data[:,:]

  selected = pca.transform(selected.reshape(1, -1))

  distances = np.linalg.norm(train_data - selected, axis=1) # to compute euclidean distance

  nearest_neighbors_ids = distances.argsort()[:k] # to find nearest two neighbors
  if return_full:
    result = {'result':tuple(original_data_all[:,1:][nearest_neighbors_ids][:k]), 'pca':pca}
  else:
    result = tuple(original_data_all[:,0][nearest_neighbors_ids][:k])
  return result