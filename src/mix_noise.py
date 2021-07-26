import glob
import math
import numpy as np
import pandas as pd
import librosa
import random
from pydub import AudioSegment
from scipy.io.wavfile import write, read
from scipy.signal import butter, lfilter, resample

def get_noise_from_sound(signal,noise,SNR):
    
    RMS_s=math.sqrt(np.mean(signal**2))

    #required RMS of noise
    RMS_n=math.sqrt(RMS_s**2/(pow(10,SNR/10)))
    
    #current RMS of noise
    RMS_n_current=math.sqrt(np.mean(noise**2))
    noise=noise*(RMS_n/RMS_n_current)
    
    return noise

data = glob.glob("../data_trimmed/*")
noiseFiles = glob.glob("../noise/*.wav")

for x in data:
    className = x.split("\\")[-1]
    print(className)
    audiofiles = glob.glob("../data_trimmed/" + className + "/*.wav")
    for y in audiofiles:
        audio, sr = librosa.load(y, sr = 8000, )
        fileName = y.split("\\")[-1].split(".")[0]
        outFile = "../data_added_noise/" + className + "/" + fileName + ".wav"
        write(outFile, sr, audio)
        for z in noiseFiles:
            noise, _ = librosa.load(z, sr = 8000)
            noiseName = z.split("\\")[-1].split(".")[0]
            randomPos = random.randint(0, len(noise)-len(audio))
            newNoise = get_noise_from_sound(audio, noise[randomPos:len(audio)+randomPos], 5)
            outFile = "../data_added_noise/" + className + "/" + fileName + "_" + noiseName + ".wav"
            write(outFile, 8000, audio + newNoise)
