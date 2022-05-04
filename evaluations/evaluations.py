import numpy as np

def cosineSimilarity(actual, predicted):
    return np.dot(actual, predicted)/(np.linalg.norm(actual)*np.linalg.norm(predicted))

def calculateRMSE(targets, predictions):
    return np.sqrt(((predictions - targets) ** 2).mean())

def calculateMAE(targets, predictions):
     return np.mean(np.abs(targets - predictions))

def getKDictionary(targets, predictions):
    kDict = {
        1: np.zeros(0),
        2: np.zeros(0),
        3: np.zeros(0),
        4: np.zeros(0),
        5: np.zeros(0),
        6: np.zeros(0),
        7: np.zeros(0),
        8: np.zeros(0),
        9: np.zeros(0),
        10: np.zeros(0)
    }
    spotifyDict = {
        1: np.zeros(0),
        2: np.zeros(0),
        3: np.zeros(0),
        4: np.zeros(0),
        5: np.zeros(0),
        6: np.zeros(0),
        7: np.zeros(0),
        8: np.zeros(0),
        9: np.zeros(0),
        10: np.zeros(0)
    }

    for j in range(predictions.shape[0]):
        predictionsKNN = predictions[j]
        predictionsSpotify = targets[j]
        meanedSpotifyPrediction = np.mean(predictionsSpotify, axis=0)

        sum = 0
        for i in range(predictionsKNN.shape[0]):
            sum += predictionsKNN[i]

            kDict[i+1] = np.concatenate((kDict[i+1], sum/(i+1)), axis=0)
            spotifyDict[i+1] = np.concatenate((spotifyDict[i+1], meanedSpotifyPrediction), axis=0)
    print(kDict)
    print(spotifyDict)
    totalDict = {
        1: (kDict[1], spotifyDict[1]),
        2: (kDict[2], spotifyDict[2]),
        3: (kDict[3], spotifyDict[3]),
        4: (kDict[4], spotifyDict[4]),
        5: (kDict[5], spotifyDict[5]),
        6: (kDict[6], spotifyDict[6]),
        7: (kDict[7], spotifyDict[7]),
        8: (kDict[8], spotifyDict[8]),
        9: (kDict[9], spotifyDict[9]),
        10: (kDict[10], spotifyDict[10])
    }

    return totalDict

def knnEvaluate(kDictionary):
    evalDict = {
        1: {
        "MAE": 0,
        "RMSE": 0,
        "COS": 0
        },
        2: {
        "MAE": 0,
        "RMSE": 0,
        "COS": 0
        },
        3: {
        "MAE": 0,
        "RMSE": 0,
        "COS": 0
        },
        4: {
        "MAE": 0,
        "RMSE": 0,
        "COS": 0
        },
        5: {
        "MAE": 0,
        "RMSE": 0,
        "COS": 0
        },
        6: {
        "MAE": 0,
        "RMSE": 0,
        "COS": 0
        },
        7: {
        "MAE": 0,
        "RMSE": 0,
        "COS": 0
        },
        8: {
        "MAE": 0,
        "RMSE": 0,
        "COS": 0
        },
        9: {
        "MAE": 0,
        "RMSE": 0,
        "COS": 0
        },
        10: {
        "MAE": 0,
        "RMSE": 0,
        "COS": 0
        }
    }
    for i in range(1, 11):
        evalDict[i]["RMSE"] = calculateRMSE(kDictionary[i][1], kDictionary[i][0])
        evalDict[i]["MAE"] = calculateMAE(kDictionary[i][1], kDictionary[i][0])
        predictionArray = kDictionary[i][0]
        actualArray = kDictionary[i][1]
        total = np.zeros(0)
        for j in range(predictionArray.shape[0]):
            total += cosineSimilarity(actualArray[j], predictionArray[j])
        total = total / predictionArray.shape[0]

        evalDict[i]["COS"] = total

    return evalDict

def calculateEvaluations(targets, predictions):
    kDictionary = getKDictionary(targets, predictions)
    evalDictionary = knnEvaluate(kDictionary)
    return evalDictionary

ourRecommendations = np.array([[[0.0297, 0.645, 0.63, 1.11e-05, 0.079, -7.017, 0.246, 174.663, 0.653],
[0.357, 0.566, 0.586, 0.0, 0.425, -6.741, 0.827, 174.262, 0.619],
[0.0929, 0.577, 0.745, 0.0, 0.368, -6.692, 0.186, 174.292, 0.424],
[0.107, 0.536, 0.628, 0.0, 0.459, -7.057, 0.186, 174.35, 0.467],
[0.402, 0.68, 0.553, 0.0, 0.196, -6.915, 0.25, 174.667, 0.558],
[0.103, 0.622, 0.807, 0.0, 0.306, -6.752, 0.408, 174.06, 0.892],
[0.0174, 0.625, 0.867, 1.28e-05, 0.0806, -6.761, 0.0685, 174.602, 0.591],
[0.013, 0.585, 0.792, 0.0, 0.126, -6.541, 0.398, 174.964, 0.752],
[0.354, 0.387, 0.844, 0.0, 0.315, -6.805, 0.14, 174.336, 0.595],
[0.142, 0.4, 0.664, 0.0, 0.11, -6.972, 0.0777, 174.679, 0.765]]])

spotifyRecommendations = np.array([[[ 8.58000e-02,  8.25000e-01,  6.72000e-01,  0.00000e+00,  1.23000e-01
,   -7.10800e+00,  4.18000e-01,  1.47010e+02,  2.46000e-01]
,  [ 5.75000e-01,  4.95000e-01,  7.59000e-01,  0.00000e+00,  1.70000e-01
,   -9.14000e+00,  4.68000e-01,  1.56627e+02,  5.03000e-01]
,  [ 5.34000e-03,  6.84000e-01,  6.93000e-01,  0.00000e+00,  5.06000e-01
,   -6.65100e+00,  3.39000e-02,  1.34988e+02,  1.08000e-01]
,  [ 3.20000e-02,  2.32000e-01,  7.05000e-01,  6.60000e-05,  1.85000e-01
,   -6.92800e+00,  9.97000e-02,  7.42800e+01,  3.52000e-01]
,  [ 1.22000e-02,  7.20000e-01,  4.24000e-01,  1.26000e-02,  4.53000e-01
,   -1.19640e+01,  3.87000e-01,  1.50010e+02,  2.28000e-01]
,  [ 3.67000e-01,  7.01000e-01,  4.85000e-01,  6.31000e-05,  8.52000e-02
,   -1.03050e+01,  3.64000e-01,  7.88630e+01,  4.96000e-01]
,  [ 3.18000e-01,  5.58000e-01,  8.64000e-01,  0.00000e+00,  1.23000e-01
,   -5.10000e+00,  4.30000e-01,  1.52804e+02,  4.74000e-01]
,  [ 1.56000e-01,  8.44000e-01,  5.53000e-01,  1.32000e-06,  8.16000e-02
,   -7.28400e+00,  3.56000e-01,  1.14939e+02,  5.11000e-01]
,  [ 1.46000e-01,  5.06000e-01,  6.28000e-01,  2.91000e-05,  1.35000e-01
,   -8.16700e+00,  1.75000e-01,  8.61310e+01,  3.63000e-01]
,  [ 3.43000e-01,  6.93000e-01,  7.13000e-01,  0.00000e+00,  1.02000e-01
,   -4.48900e+00,  3.51000e-01,  1.37335e+02,  6.15000e-01]]])

evalDict = calculateEvaluations(spotifyRecommendations, ourRecommendations)
print(evalDict)
