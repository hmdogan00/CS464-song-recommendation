from flask import Flask, request
from flask_cors import cross_origin
import json 
import pandas as pd
import numpy as np
import requests
from sklearn.decomposition import PCA

from models.knn import knn
from models.pca_knn import pcaknn
from models.NN import NN
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

@app.route('/nn', methods=['PUT'])
def get_nn():
    data = request.get_json()['body']

    # Data variable contains the
    # data from the node server
    result = NN(data)
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
    pca_n_2 = []
    pca_spotify_n_2 = []
    pca_n_3 = []
    pca_spotify_n_3 = []
    pca_n_4 = []
    pca_spotify_n_4 = []
    pca_n_5 = []
    pca_spotify_n_5 = []
    spoti_results = []

    nn_results = []
    for i, track in df.iterrows():
        knn_results.append(knn(track, k=10, return_full=True))
        recoms = requests.get(url=('https://api.spotify.com/v1/recommendations/?seed_tracks='+data['audio_features'][i]['id']+'&limit=1'), headers=header)
        tmp = []
        for rec in (recoms.json()['tracks']):
            feats = requests.get(url=('https://api.spotify.com/v1/audio-features?ids='+rec['id']), headers=header)
            feats = feats.json()['audio_features'][0]
            tmp.append(np.array([feats['acousticness'], feats['danceability'], feats['energy'], feats['instrumentalness'], feats['liveness'], feats['loudness'], feats['speechiness'], feats['tempo'], feats['valence']]))
        spoti_results.append(np.array(tmp, dtype=np.float64))

        pcaknn2 = pcaknn(track, k=10, return_full=True, n=2)
        res = pcaknn2['result']
        pca = pcaknn2['pca']
        pca_n_2.append(res)
        pca_spotify_n_2.append(pca.transform(np.array(tmp).reshape(1, -1)))

        pcaknn3 = pcaknn(track, k=10, return_full=True, n=3)
        res = pcaknn3['result']
        pca = pcaknn3['pca']
        pca_n_3.append(res)
        pca_spotify_n_3.append(pca.transform(np.array(tmp).reshape(1, -1)))

        pcaknn4 = pcaknn(track, k=10, return_full=True, n=4)
        res = pcaknn4['result']
        pca = pcaknn4['pca']
        pca_n_4.append(res)
        pca_spotify_n_4.append(pca.transform(np.array(tmp).reshape(1, -1)))

        pcaknn5 = pcaknn(track, k=10, return_full=True, n=5)
        res = pcaknn5['result']
        pca = pcaknn5['pca']
        pca_n_5.append(res)
        pca_spotify_n_5.append(pca.transform(np.array(tmp).reshape(1, -1)))

        nn_results.append(NN(track, return_full=True))

        print(i+1)
            
    knn_results = np.array(knn_results, dtype=np.float64)
    spoti_results = np.array(spoti_results, dtype=np.float64)
    pca_n_2 = np.array(pca_n_2, dtype=np.float64)
    pca_spotify_n_2 = np.array(pca_spotify_n_2, dtype=np.float64)
    pca_n_3 = np.array(pca_n_3, dtype=np.float64)
    pca_spotify_n_3 = np.array(pca_spotify_n_3, dtype=np.float64)
    pca_n_4 = np.array(pca_n_4, dtype=np.float64)
    pca_spotify_n_4 = np.array(pca_spotify_n_4, dtype=np.float64)
    pca_n_5 = np.array(pca_n_5, dtype=np.float64)
    pca_spotify_n_5 = np.array(pca_spotify_n_5, dtype=np.float64)
    nn_results = np.array(nn_results, dtype=np.float64)
    
    for i in range(knn_results.shape[0]):
        for j in range(knn_results.shape[2]):
            col = knn_results[i][:,j]
            knn_results[i][:,j] = (col - col.min()) / (col.max() - col.min())
            col3 = nn_results[i][:,j]
            nn_results[i][:,j] = (col3 - col3.min()) / (col3.max() - col3.min())
    eval_dict = calculateEvaluations(spoti_results, knn_results)
    eval_dict_pca2 = calculateEvaluations(pca_spotify_n_2, pca_n_2)
    eval_dict_pca3 = calculateEvaluations(pca_spotify_n_3, pca_n_3)
    eval_dict_pca4 = calculateEvaluations(pca_spotify_n_4, pca_n_4)
    eval_dict_pca5 = calculateEvaluations(pca_spotify_n_5, pca_n_5)
    eval_dict_nn = calculateEvaluations(spoti_results, nn_results)
    
    df = pd.DataFrame(eval_dict_nn).T.values
    print(df)
    df_pca2 = pd.DataFrame(eval_dict_pca2).T.values
    df_pca3 = pd.DataFrame(eval_dict_pca3).T.values
    df_pca4 = pd.DataFrame(eval_dict_pca4).T.values
    df_pca5 = pd.DataFrame(eval_dict_pca5).T.values

    b_mae = int(df[:,0].argmin()+1)
    b_rmse = int(df[:,1].argmin()+1)
    b_cos_i = int(df[:,2].argmax()+1)
    b_cos_v = int(df[df[:,2].argmax(),2])
    
    w_mae = int(df[:,0].argmax()+1)
    w_rmse = int(df[:,1].argmax()+1)
    w_cos_i = int(df[:,2].argmin()+1)
    w_cos_v = int(df[df[:,2].argmin(),2])

    print('FEATURE EXTRACTION - NEURAL NETWORK')
    print('Best k parameters for different metrics:')
    print('MAE: k=',df[:,0].argmin()+1, ' Value=', df[df[:,0].argmax(),0])
    print('RMSE: k=',df[:,1].argmin()+1, ' Value=', df[df[:,1].argmax(),1])
    print('Cosine Similarity: k=',df[:,2].argmax()+1, ' Value=', df[df[:,2].argmax(),2])

    """ 
    print('ONLY KNN')
    print('Best k parameters for different metrics:')
    print('MAE: k=',df[:,0].argmin()+1, ' Value=', df[df[:,0].argmax(),0])
    print('RMSE: k=',df[:,1].argmin()+1, ' Value=', df[df[:,1].argmax(),1])
    print('Cosine Similarity: k=',df[:,2].argmax()+1, ' Value=', df[df[:,2].argmax(),2])
    
    print('Worst k parameters for different metrics:')
    print('MAE: k=',df[:,0].argmax()+1, ' Value=', df[df[:,0].argmax(),0])
    print('RMSE: k=',df[:,1].argmax()+1, ' Value=', df[df[:,1].argmax(),1])
    print('Cosine Similarity: k=',df[:,2].argmin()+1, ' Value=', df[df[:,2].argmin(),2])
    print('\n\nPCA with N=2 + KNN:')
    print('Best k parameters for different metrics:')
    print('MAE: k=',df_pca2[:,0].argmin()+1, ' Value=', df_pca2[df_pca2[:,0].argmax(),0])
    print('RMSE: k=',df_pca2[:,1].argmin()+1, ' Value=', df_pca2[df_pca2[:,1].argmax(),1])
    print('Cosine Similarity: k=',df_pca2[:,2].argmax()+1, ' Value=', df_pca2[df_pca2[:,2].argmax(),2])
    
    print('Worst k parameters for different metrics:')
    print('MAE: k=',df_pca2[:,0].argmax()+1, ' Value=', df_pca2[df_pca2[:,0].argmax(),0])
    print('RMSE: k=',df_pca2[:,1].argmax()+1, ' Value=', df_pca2[df_pca2[:,1].argmax(),1])
    print('Cosine Similarity: k=',df_pca2[:,2].argmin()+1, ' Value=', df_pca2[df_pca2[:,2].argmin(),2])

    print('\n\nPCA with N=3 + KNN:')
    print('Best k parameters for different metrics:')
    print('MAE: k=',df_pca3[:,0].argmin()+1, ' Value=', df_pca3[df_pca3[:,0].argmax(),0])
    print('RMSE: k=',df_pca3[:,1].argmin()+1, ' Value=', df_pca3[df_pca3[:,1].argmax(),1])
    print('Cosine Similarity: k=',df_pca3[:,2].argmax()+1, ' Value=', df_pca3[df_pca3[:,2].argmax(),2])
    
    print('Worst k parameters for different metrics:')
    print('MAE: k=',df_pca3[:,0].argmax()+1, ' Value=', df_pca3[df_pca3[:,0].argmax(),0])
    print('RMSE: k=',df_pca3[:,1].argmax()+1, ' Value=', df_pca3[df_pca3[:,1].argmax(),1])
    print('Cosine Similarity: k=',df_pca3[:,2].argmin()+1, ' Value=', df_pca3[df_pca3[:,2].argmin(),2])

    print('\n\nPCA with N=4 + KNN:')
    print('Best k parameters for different metrics:')
    print('MAE: k=',df_pca4[:,0].argmin()+1, ' Value=', df_pca4[df_pca4[:,0].argmax(),0])
    print('RMSE: k=',df_pca4[:,1].argmin()+1, ' Value=', df_pca4[df_pca4[:,1].argmax(),1])
    print('Cosine Similarity: k=',df_pca4[:,2].argmax()+1, ' Value=', df_pca4[df_pca4[:,2].argmax(),2])
    
    print('Worst k parameters for different metrics:')
    print('MAE: k=',df_pca4[:,0].argmax()+1, ' Value=', df_pca4[df_pca4[:,0].argmax(),0])
    print('RMSE: k=',df_pca4[:,1].argmax()+1, ' Value=', df_pca4[df_pca4[:,1].argmax(),1])
    print('Cosine Similarity: k=',df_pca4[:,2].argmin()+1, ' Value=', df_pca4[df_pca4[:,2].argmin(),2])

    print('\n\nPCA with N=5 + KNN:')
    print('Best k parameters for different metrics:')
    print('MAE: k=',df_pca5[:,0].argmin()+1, ' Value=', df_pca5[df_pca5[:,0].argmax(),0])
    print('RMSE: k=',df_pca5[:,1].argmin()+1, ' Value=', df_pca5[df_pca5[:,1].argmax(),1])
    print('Cosine Similarity: k=',df_pca5[:,2].argmax()+1, ' Value=', df_pca5[df_pca5[:,2].argmax(),2])
    
    print('Worst k parameters for different metrics:')
    print('MAE: k=',df_pca5[:,0].argmax()+1, ' Value=', df_pca5[df_pca5[:,0].argmax(),0])
    print('RMSE: k=',df_pca5[:,1].argmax()+1, ' Value=', df_pca5[df_pca5[:,1].argmax(),1])
    print('Cosine Similarity: k=',df_pca5[:,2].argmin()+1, ' Value=', df_pca5[df_pca5[:,2].argmin(),2]) """
    # Return data in json format
    return json.dumps({"best": {'mae':b_mae, 'rmse':b_rmse, 'cos_index':b_cos_i, 'cos_value':b_cos_v}, 
                "worst":{'mae':w_mae, 'rmse':w_rmse, 'cos_index':w_cos_i, 'cos_value':w_cos_v}})
    
   
if __name__ == "__main__": 
    app.run(port=5000)