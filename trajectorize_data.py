import numpy as np
import librosa
import glob
import pickle
from tqdm import tqdm

from MusicVectorizer import MusicVectorizer

SAMPLE_RATE = 16000

data_folders = ["90s_hiphop", "90s_rock", "2010s_pop", "classical", "country"]

mv = MusicVectorizer()

for folder in data_folders[1:]:
    data = pickle.load(open("data/" + folder + ".pkl", "rb"))
    print(folder)
    trajectories = dict()
    for path, song in tqdm(data.items()):
        try:
            trajectories[path] = mv.trajectorize_song(song, SAMPLE_RATE)
        except:
            print("Error vectorizing file: ", path)
    np.save("data/" + folder + "_trajectories.npy", trajectories)