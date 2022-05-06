from flask import Flask, request
from flask_cors import cross_origin
import json 
import pandas as pd
import numpy as np
import requests

from models.knn import knn
from models.pca_knn import pcaknn
from evaluations.evaluations import calculateEvaluations

app = Flask(__name__) 
  
# Setup url route which will calculate
# total sum of array.
@app.route('/knn', methods = ['PUT'])
def get_knn():
    data = request.get_json()['body']
    k = request.get_json()['k']
    metric = request.get_json()['metric']
    # Data variable contains the
    # data from the node server
    result = knn(data,k=int(k), dist_metric=metric)
    # Return data in json format
    return json.dumps({"result":result})


@app.route('/pcaknn', methods=['PUT'])
def get_pcaknn():
    data = request.get_json()['body']

    # Data variable contains the
    # data from the node server
    result = pcaknn(data)
    # Return data in json format
    return json.dumps({"result": result})

@app.route('/eval', methods=['POST'])
@cross_origin()
def get_evaluation():
    ids = request.form['ids']
    token = request.form['token']
    header = {"Content-Type":"application/json", "Authorization": "Bearer " + token}

    # get features of taken data
    r = requests.get(url=('https://api.spotify.com/v1/audio-features?ids='+ids), headers=header)
    data = r.json()
    df = pd.DataFrame(data['audio_features'])
    # get knn and spotify results
    knn_results = []
    spoti_results = []
    for i, track in df.iterrows():
        knn_results.append(knn(track, k=10, return_full=True))
        recoms = requests.get(url=('https://api.spotify.com/v1/recommendations/?seed_tracks='+data['audio_features'][i]['id']+'&limit=1'), headers=header)
        tmp = []
        for rec in (recoms.json()['tracks']):
            feats = requests.get(url=('https://api.spotify.com/v1/audio-features?ids='+rec['id']), headers=header)
            feats = feats.json()['audio_features'][0]
            tmp.append(np.array([feats['acousticness'], feats['danceability'], feats['energy'], feats['instrumentalness'], feats['liveness'], feats['loudness'], feats['speechiness'], feats['tempo'], feats['valence']]))
        spoti_results.append(np.array(tmp, dtype=np.float64))
        if i == 25:
            print('25%.....')
        elif i == 50:
            print('50%.....')
        elif i == 75:
            print('75%.....')
            
    knn_results = np.array(knn_results, dtype=np.float64)
    spoti_results = np.array(spoti_results, dtype=np.float64)
    print(spoti_results.shape)
    
    for i in range(knn_results.shape[0]):
        for j in range(knn_results.shape[2]):
            col = knn_results[i][:,j]
            knn_results[i][:,j] = (col - col.min()) / (col.max() - col.min())
    eval_dict = calculateEvaluations(spoti_results, knn_results)
    print(eval_dict)
    df = pd.DataFrame(eval_dict).T.values
    b_mae = df[:,0].argmin()+1
    b_rmse = df[:,1].argmin()+1
    b_cos_i = df[:,2].argmax()+1
    b_cos_v = df[df[:,2].argmax(),2]
    
    w_mae = df[:,0].argmax()+1
    w_rmse = df[:,1].argmax()+1
    w_cos_i = df[:,2].argmin()+1
    w_cos_v = df[df[:,2].argmin(),2]
    # Return data in json format
    return json.dumps({"best": {'mae':b_mae, 'rmse':b_rmse, 'cos_index':b_cos_i, 'cos_value':b_cos_v}, 
                "worst":{'mae':w_mae, 'rmse':w_rmse, 'cos_index':w_cos_i, 'cos_value':w_cos_v}})
    
   
if __name__ == "__main__": 
    app.run(port=5000)