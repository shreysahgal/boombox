import numpy as np
import librosa
from glob import glob
import pickle
from tqdm import tqdm
import os
import pickletools
import pandas as pd
from MusicVectorizer import MusicVectorizer
from train_encoding_model import BoomboxNet
import torch
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

SAMPLE_RATE = 16000

data_folder = "data/large_dataset"

mv = MusicVectorizer()

encoding_model = BoomboxNet()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoding_model.load_state_dict(torch.load("models/model_50000.pt", map_location=torch.device(device)))

genres = [os.path.basename(folder) for folder in glob(data_folder + "/*") if "." not in folder]

df = pd.DataFrame(columns=["genre", "file", "trajectory"])

totals, counts = [], []


SAMPLE_TIME = 3
NUM_SONGLETS = 10
NORM = True

for genre in genres:
    print(genre)
    trajectories = dict()

    totals.append(0)
    counts.append(0)

    for file in tqdm(glob(f"{data_folder}/{genre}/*.mp3")):
        y, sr = librosa.load(file, sr=SAMPLE_RATE)
        totals[-1] += 1

        try:
            traj = mv.trajectorize_song(y, SAMPLE_RATE, sample_time=SAMPLE_TIME)
            traj = encoding_model.fc1(torch.tensor(traj).flatten(start_dim=1).float()).detach().numpy()
            pca = PCA(n_components=2).fit(traj)
            reduced = pca.transform(traj)

            df = pd.concat([df, pd.DataFrame({"genre": genre, "file": file, "trajectory": [np.float32(reduced)]})], ignore_index=True)
        
        except Exception as e:
            print(f"Error with {file}: {e}")
            continue

df.to_pickle(f"{data_folder}/viz_trajectories.pkl")

for i in range(len(genres)):
    print(f"{genres[i]}: {counts[i]}/{totals[i]}")

breakpoint()