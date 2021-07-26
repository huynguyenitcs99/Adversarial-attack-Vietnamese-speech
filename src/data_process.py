import glob
import math
import random
import librosa
import numpy as np
import tensorflow.keras
from scipy.io.wavfile import write

def split_data(data):
    numVal = len(data)//10
    numTest = 2*len(data)//10
    train = data
    val = []
    test = []
    for i in range(numVal):
        randomId = random.randint(0, len(train)-1)
        val += [train.pop(randomId)]
    for i in range(numTest):
        randomId = random.randint(0, len(train)-1)
        test += [train.pop(randomId)]
    return train, val, test

def load_data():
    trainData = []
    valData = []
    testData = []

    classDirs = glob.glob("../data_added_noise/*")
    classDict = {'Close_door': 0, 'Close_gate': 1, 'Doraemon': 2, 'Lock_door': 3, 'Lock_gate': 4, 'Open_door': 5, 'Open_gate': 6, 'Turn_off_air_conditioner': 7, 'Turn_off_fan': 8, 'Turn_off_light': 9, 'Turn_off_TV': 10, 'Turn_on_air_conditioner': 11, 'Turn_on_fan': 12, 'Turn_on_light': 13, 'Turn_on_TV': 14}

    for x in classDirs:
        className = x.replace("\\","/").split("/")[-1]
        temp = []
        files = glob.glob(x + "/*.wav")
        for y in files:
            audio, sr = librosa.load(y, sr = 8000)
            temp.append((audio, classDict[className]))

        trainTemp, valTemp, testTemp = split_data(temp)
        trainData += trainTemp
        valData += valTemp
        testData += testTemp

    return trainData, valData, testData

class SpeechGen(tensorflow.keras.utils.Sequence):

    def __init__(self, data_In, batch_size=64,
                 dim=8000, shuffle=True):

        self.dim = dim*2
        self.batch_size = batch_size
        self.data_In = data_In
        self.shuffle = shuffle
        self.data_indexes = np.arange(len(self.data_In))
        self.on_epoch_end()

    def __len__(self):

        return int(np.floor(len(self.data_indexes) / self.batch_size))

    def __getitem__(self, index):

        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        list_IDs_temp = [self.data_indexes[k] for k in indexes]

        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):

        self.indexes = np.arange(len(self.data_In))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.dim))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):

            # load data from file, saved as numpy array on disk
            curX = self.data_In[ID][0]

            # normalize
            # invMax = 1/(np.max(np.abs(curX))+1e-3)
            # curX *= invMax

            if curX.shape[0] == self.dim:
                X[i] = curX
            else:
                randPos = np.random.randint(self.dim-curX.shape[0])
                X[i, randPos:randPos + curX.shape[0]] = curX

            # Store class
            y[i] = self.data_In[ID][1]

        return X, y