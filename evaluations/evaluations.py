import numpy as np

def cosineSimilarity(actual, predicted):
    return np.dot(actual, predicted)/(np.linalg.norm(actual)*np.linalg.norm(predicted))

def calculateRMSE(targets, predictions):
    return np.sqrt(((predictions - targets) ** 2).mean())

def calculateMAE(targets, predictions):
     return np.mean(np.abs(targets - predictions))

def getKDictionary(allSongsWithPredictions):
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

    for song in allSongsWithPredictions:
        predictionsKNN = song[0]
        predictionsSpotify = song[1]
        meanedSpotifyPrediction = np.mean(predictionsSpotify, axis=0)

        sum = 0
        for i in range(predictionsKNN):
            sum += predictionsKNN[i]





def calculateEvaluations(allSongsWithPredictions):
    