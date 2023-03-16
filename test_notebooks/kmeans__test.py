#!/usr/bin/env python

"""
Idea: if the MAP-MERT model captures useful information about the music, then
we should be able to cluster songs based on their feature vectors.

Test: data/classical and data/90s_hiphop are two folders with classical and 90s
hiphop songs. We will get the feature vectors for each song, and then cluster,
and see if the clusters make sense.

Note: the trajectorized songs will be different lengths, since each song is a
different length. Here, we distance metric we use for clustering is to take the
'endpoint' of the song. That is, we treat the feature vectors as a trajectory,
so to get the final location of a point starting at the origin and following
the feature trajectory, we get the sum of all the columns of the feature
vectors at each timestep.
"""

import librosa
from MusicVectorizer import MusicVectorizer
import numpy as np
import glob
from tqdm import tqdm
from sklearn.cluster import KMeans

SAMPLE_RATE = 16000  # this needs to be hardcoded for the pretrained hubert model

# Load in the data
classical_songs = glob.glob("data/classical/*.mp3")
hiphop_songs = glob.glob("data/90s_hiphop/*.mp3")

# Vectorize the songs
vectorizer = MusicVectorizer()

classical_vectors = []
hiphop_vectors = []

for song in tqdm(classical_songs):
    y = librosa.load(song, sr=SAMPLE_RATE)
    classical_vectors.append(vectorizer.trajectorize_song(y, SAMPLE_RATE))

for song in tqdm(hiphop_songs):
    y = librosa.load(song, sr=SAMPLE_RATE)
    hiphop_vectors.append(vectorizer.trajectorize_song(y, SAMPLE_RATE))

# Cluster the songs
classical_endpoints = np.array([np.sum(song, axis=0) for song in classical_vectors])
hiphop_endpoints = np.array([np.sum(song, axis=0) for song in hiphop_vectors])

# concatenate the first 80% of each endpoints list to get the train_data
# and the last 20% of each endpoints list to get the test_data
train_data = np.concatenate((classical_endpoints[:int(len(classical_endpoints) * 0.8)], hiphop_endpoints[:int(len(hiphop_endpoints) * 0.8)]))
test_data = np.concatenate((classical_endpoints[int(len(classical_endpoints) * 0.8):], hiphop_endpoints[int(len(hiphop_endpoints) * 0.8):]))

# train kmeans model
model = KMeans(n_clusters=2, random_state=0).fit(train_data)

# test model
predictions = model.predict(test_data)
# if the model is good, then the first half of the test data should be in one cluster and 
# the second half should be in the other cluster
print(predictions)

breakpoint()