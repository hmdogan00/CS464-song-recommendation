import numpy as np
import sys
import pandas as pd
import json
import os
from sklearn.processing import StandardScaler
from sklearn.decomposition import PCA
dirname = os.path.dirname(__file__)
location = os.path.join(dirname, '../data/SpotifyFeatures.csv')
selected = -1
id = -1
if len(sys.argv) > 1:
  dict = json.loads(sys.argv[1])
  selected = np.array([dict['acousticness'], dict['danceability'], dict['energy'], dict['instrumentalness'], dict['liveness'], dict['loudness'], dict['speechiness'], dict['tempo'], dict['valence']], dtype='float64')
  id = dict['id']

data = pd.read_csv(location)
data = data.drop_duplicates(subset=['track_id'])
data = data[data['track_id'] != id]

features_all = data[["genre","artist_name","track_name","track_id", "acousticness", "danceability","energy", "instrumentalness", "liveness", "loudness", "speechiness", "tempo", "valence"]]
#apply PCA
X = data[features_all].values
#standardize data befor PCA (normalize data)
X = StandardScaler().fit_transform(X)
#create PCA
pca = PCA(n_components=6) #look later
princ_comps = pca.fit_transform(X)
#new dataset by using principle components
new_data = pd.DataFrame(data = princ_comps, columns=['PC1','PC2','PC3','PC4','PC5','PC6'])
new_features_used = new_data[['PC1','PC2','PC3','PC4','PC5','PC6']]
original_data_all = np.array(new_features_used.values)

np.random.seed(1)
np.random.shuffle(original_data_all)

original_data1 = original_data_all[:,4:]

original_data = np.array(original_data1,"float64")


train_data = original_data[:141420,:]

test_data = original_data[141421:,:]

track_number = 0
track_selected = test_data[track_number,:]

distances = np.linalg.norm(train_data - selected, axis=1) # to compute euclidean distance

k = 2

nearest_neighbors_ids = distances.argsort()[:k] # to find nearest two neighbors

print(original_data_all[:,3][nearest_neighbors_ids][0], original_data_all[:,3][nearest_neighbors_ids][1])
sys.stdout.flush()