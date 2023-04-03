import numpy as np
import librosa
import glob
import os
import pickle
from tqdm import tqdm

from MusicVectorizer import MusicVectorizer

SAMPLE_RATE = 16000

data_folder = "/home/shrey/Documents/eecs448-boombox/data/gtzan/"
genres = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]

mv = MusicVectorizer()

for genre in genres:
    folder = data_folder + genre
    files = glob.glob(folder + "/*.wav")
    trajs = dict()

    for file in tqdm(files):
        song, _ = librosa.load(file, sr=SAMPLE_RATE)
        traj = mv.trajectorize_song(song, SAMPLE_RATE)
        trajs[os.path.basename(file)] = traj
    
    with open(f"{folder}/trajs.pkl", "wb") as f:
        pickle.dump(trajs, f)
