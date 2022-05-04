import numpy as np
import pandas as pd

def cosineSimilarity(actual, predicted):
    return np.dot(actual, predicted)/(np.linalg.norm(actual)*np.linalg.norm(predicted))

def calculateRMSE(targets, predictions):
    return np.sqrt(((predictions - targets) ** 2).mean())

def calculateMAE(targets, predictions):
     return np.mean(np.abs(targets - predictions))

def getKDictionary(targets, predictions):
    kDict = {
        1: np.zeros((predictions.shape[0],predictions.shape[2])),
        2: np.zeros((predictions.shape[0],predictions.shape[2])),
        3: np.zeros((predictions.shape[0],predictions.shape[2])),
        4: np.zeros((predictions.shape[0],predictions.shape[2])),
        5: np.zeros((predictions.shape[0],predictions.shape[2])),
        6: np.zeros((predictions.shape[0],predictions.shape[2])),
        7: np.zeros((predictions.shape[0],predictions.shape[2])),
        8: np.zeros((predictions.shape[0],predictions.shape[2])),
        9: np.zeros((predictions.shape[0],predictions.shape[2])),
        10: np.zeros((predictions.shape[0],predictions.shape[2])),
    }

    for j in range(predictions.shape[0]):
        predictionsKNN = predictions[j]
        spotifyPrediction = targets[j]

        sum = np.zeros(predictionsKNN.shape[1])
        for i in range(predictionsKNN.shape[0]):
            sum += predictionsKNN[i]
            kDict[i+1][j] = sum/(i+1)

    return kDict, spotifyPrediction

def knnEvaluate(kDictionary, spotifyPred):
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
        evalDict[i]["RMSE"] = calculateRMSE(spotifyPred, kDictionary[i])
        evalDict[i]["MAE"] = calculateMAE(spotifyPred, kDictionary[i])
        predictionArray = kDictionary[i]
        actualArray = spotifyPred
        total = 0
        for j in range(predictionArray.shape[0]):
            total += cosineSimilarity(actualArray, predictionArray[j])
        total = total / predictionArray.shape[0]

        evalDict[i]["COS"] = total

    return evalDict

def calculateEvaluations(targets, predictions):
    kDictionary, spotifyPred = getKDictionary(targets, predictions)
    evalDictionary = knnEvaluate(kDictionary, spotifyPred)
    return evalDictionary

if __name__ == '__main__':
    ourRecommendations = np.array([[[0.0297, 0.645, 0.63, 1.11e-05, 0.079, -7.017, 0.246, 174.663, 0.653],
    [0.357, 0.566, 0.586, 0.0, 0.425, -6.741, 0.827, 174.262, 0.619],
    [0.0929, 0.577, 0.745, 0.0, 0.368, -6.692, 0.186, 174.292, 0.424],
    [0.107, 0.536, 0.628, 0.0, 0.459, -7.057, 0.186, 174.35, 0.467],
    [0.402, 0.68, 0.553, 0.0, 0.196, -6.915, 0.25, 174.667, 0.558],
    [0.103, 0.622, 0.807, 0.0, 0.306, -6.752, 0.408, 174.06, 0.892],
    [0.0174, 0.625, 0.867, 1.28e-05, 0.0806, -6.761, 0.0685, 174.602, 0.591],
    [0.013, 0.585, 0.792, 0.0, 0.126, -6.541, 0.398, 174.964, 0.752],
    [0.354, 0.387, 0.844, 0.0, 0.315, -6.805, 0.14, 174.336, 0.595],
    [0.142, 0.4, 0.664, 0.0, 0.11, -6.972, 0.0777, 174.679, 0.765]],

    [[0.0297, 0.645, 0.63, 1.11e-05, 0.079, -7.017, 0.246, 174.663, 0.653],
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
    ,   -7.10800e+00,  4.18000e-01,  1.47010e+02,  2.46000e-01]],

    [[ 8.58000e-02,  8.25000e-01,  6.72000e-01,  0.00000e+00,  1.23000e-01
    ,   -7.10800e+00,  4.18000e-01,  1.47010e+02,  2.46000e-01]]])

    evalDict = calculateEvaluations(spotifyRecommendations, ourRecommendations)
    df = pd.DataFrame(evalDict).T
    print(df[["MAE"]].idxmin())
    print(df[["RMSE"]])
    print(df)
