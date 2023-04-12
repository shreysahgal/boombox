import pandas as pd
import numpy as np
import torch

class BoomboxRecommender:
    def recommend_songs(self, liked_songs, all_songs, S=10, return_num=10):
        projections = list()
        for songlet in range(S):
            A = liked_songs[songlet].T
            P = A @ np.linalg.inv((A.T  @ A)) @ A.T
            projections.append(P)

        projected_songs = np.zeros_like(all_songs)
        for song in range(all_songs.shape[0]):
            for songlet in range(S):
                proj = projections[songlet] @ all_songs[song, songlet, :]
                projected_songs[song, songlet, :] = proj
        
        norms = torch.norm((torch.tensor(projected_songs) - torch.tensor(trajs)), dim=2, p=2)
        norms = torch.sum(norms, dim=1)
        _, indices = torch.sort(norms, descending=False)

        return indices[:return_num]
    
if __name__ == "__main__":

    df = pd.read_pickle("data/large_dataset/correct_trajectories.pkl")
    label_dict = {genre: idx for idx, genre in enumerate(df["genre"].unique())}
    df["name"] = df["file"].apply(lambda x: x.split("/")[-1])

    trajs = np.concatenate(df["trajectory"].to_numpy()).reshape(-1, 10, 768)
    names = df["name"].to_numpy()
    genres = df["genre"].to_numpy()
    labels = np.array([label_dict[genre] for genre in genres])

    liked_names = [
        "data/large_dataset/80s_pop/Culture Club - Karma Chameleon (Official Music Video).mp3",
        "data/large_dataset/80s_pop/Daryl Hall & John Oates - You Make My Dreams (Official HD Video).mp3",
        "data/large_dataset/90s_hiphop/Wu-Tang Clan - C.R.E.A.M. (Official HD Video).mp3",
        "data/large_dataset/2000s_alt_rock/MGMT - Electric Feel (Official HD Video).mp3",
        "data/large_dataset/2000s_alt_rock/Red Hot Chili Peppers - Californication (Official Music Video) [HD UPGRADE].mp3",
        "data/large_dataset/2000s_alt_rock/Green Day - 21 Guns [Official Music Video].mp3",
        "data/large_dataset/2000s_rnb/Dr. Dre - Still D.R.E. ft. Snoop Dogg.mp3",
        "data/large_dataset/2010s_hiphop/Outkast - Hey Ya! (Official HD Video).mp3",
        "data/large_dataset/2010s_pop/The Weeknd - I Feel It Coming ft. Daft Punk (Official Video).mp3",
        "data/large_dataset/2010s_pop/Icona Pop - I Love It (feat. Charli XCX) [OFFICIAL VIDEO].mp3"
    ]

    # get all the trajectories for the liked songs from df
    liked = np.zeros((10, 10, 768))
    for i, name in enumerate(liked_names):
        liked[i] = df[df["file"] == name]["trajectory"].iloc[0]
    
    # get recommendations
    bbr = BoomboxRecommender()
    indices = bbr.recommend_songs(liked, trajs)

    # print names of recommendations
    print(names[indices])