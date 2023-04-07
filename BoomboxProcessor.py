import numpy as np
from tqdm import tqdm
import torch
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
import os

class BoomboxProcessor:
    def __init__(self, verbose:bool=False) -> None:
        self.verbose : bool = verbose

    def load_trajectories(self, data_folder : str, genres : list[str], traj_file : str, is_numpy : bool = False):
        """
        Loads the trajectories from the given data folders.
        """
        self.data_folder = data_folder
        self.genres : list[str] = genres
        self.trajectories : dict[str, dict[str, np.ndarray]]= dict()
        self.num_trajectories : dict[str, int] = dict()
        self.song_lengths = dict()

        for genre in self.genres:
            file = os.path.join(self.data_folder, genre, traj_file)
            if not is_numpy:
                self.trajectories[genre] = np.load(file, allow_pickle=True)
            else:
                self.trajectories[genre] = np.load(file, allow_pickle=True).item()
            self.num_trajectories[genre] = len(self.trajectories[genre])
            self.song_lengths[genre] = {song: trajectory.shape[0] for song, trajectory in self.trajectories[genre].items()}
            if self.verbose:
                print(f"Loaded {self.num_trajectories[genre]} trajectories from {genre}")
        
        self.labels = dict()
        for idx, genre in enumerate(self.genres):
            self.labels[genre] = idx


    def get_labels(self):
        """
        Returns a dictionary of the (int) labels for each genre.
        """
        return self.labels
    
    def get_num_trajectories(self):
        """
        Returns a dictionary of the number of trajectories for each genre.
        """
        return self.num_trajectories

    def get_trajectories(self):
        return self.trajectories
    
    def get_genre_trajectories(self, genre:str) -> tuple[list[np.ndarray], list[int]]:
        """
        Returns a list of all the trajectories for a given genre as well as the genre label.
        """
        if genre in self.genres:
            return (list(self.trajectories[genre].values()), [self.labels[genre]]*self.num_trajectories[genre])
        else:
            raise Exception("Genre not found")
    
    def get_all_trajectories(self) -> tuple[list[np.ndarray], list[int]]:
        """
        Returns a list of all the trajectories as well as a list all the genre labels.
        """
        all_trajectories = []
        all_labels = []
        for genre in self.genres:
            all_trajectories += list(self.trajectories[genre].values())
            all_labels += [self.labels[genre]]*self.num_trajectories[genre]

        return (all_trajectories, all_labels)

    def get_all_features(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns a list of all the features as well as a list all the genre labels.
        """
        N = sum([sum(self.song_lengths[folder].values()) for folder in self.genres])
        all_features = np.zeros((N, 13, 768))
        all_labels = np.zeros(N, dtype=int)

        i = 0
        for genre in self.genres:
            for file in self.trajectories[genre]:
                all_features[i:i+self.song_lengths[genre][file]] = self.trajectories[genre][file]
                all_labels[i:i+self.num_trajectories[genre]] = self.labels[genre]
                i += self.song_lengths[genre][file]

        return (all_features, all_labels)
    
    def load_encoding_model(self, model_path:str, ModelClass: torch.nn.Module) -> None:
        """
        Loads the encoding model from the given path.
        """
        self.encoding_model = ModelClass()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoding_model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    
    def encode_trajectories(self, device='cpu') -> None:
        """
        Encodes all the trajectories using the encoding model.
        """
        if self.encoding_model is None:
            raise Exception("Encoding model not loaded")

        self.encoded_trajectories = dict()
        for genre in self.genres:
            self.encoded_trajectories[genre] = dict()
            iterator = tqdm(self.trajectories[genre]) if self.verbose else self.trajectories[genre]
            for file in iterator:
                new_traj = self.encoding_model.fc1(
                    torch.tensor(self.trajectories[genre][file]).flatten(start_dim=1).float()
                ).detach().numpy()
                self.encoded_trajectories[genre][file] = new_traj.reshape(self.trajectories[genre][file].shape[0], 768)

    def encode_trajectories_pca(self) -> None:
        """
        Encodes all the trajectories using PCA.
        """
        self.encoded_trajectories = dict()
        for genre in self.genres:
            self.encoded_trajectories[genre] = dict()
            genres_iter = tqdm(self.trajectories[genre]) if self.verbose else self.trajectories[genre]
            for song in genres_iter:
                pca = PCA(n_components=1)
                pca.fit(self.trajectories[genre][song].reshape(-1, 13))
                self.encoded_trajectories[genre][song] = \
                    pca.transform(self.trajectories[genre][song].reshape(-1, 13)).reshape(self.trajectories[genre][song].shape[0], 768)
        
    def get_encoded_trajectories(self):
        if self.encoded_trajectories is None:
            raise Exception("Trajectories not encoded")
        return self.encoded_trajectories
    
    def get_encoded_genre_trajectories(self, genre:str) -> tuple[list[np.ndarray], list[int]]:
        """
        Returns a list of all the encoded trajectories for a given genre as well as the genre label.
        """
        if self.encoded_trajectories is None:
            raise Exception("Trajectories not encoded")
        if genre in self.genres:
            return (list(self.encoded_trajectories[genre].values()), [self.labels[genre]]*self.num_trajectories[genre])
        else:
            raise Exception("Genre not found")
    
    def get_all_encoded_trajectories(self) -> tuple[list[np.ndarray], list[int]]:
        """
        Returns a list of all the encoded trajectories as well as a list all the genre labels.
        """
        if self.encoded_trajectories is None:
            raise Exception("Trajectories not encoded")
        all_trajectories = []
        all_labels = []
        for genre in self.genres:
            all_trajectories += list(self.encoded_trajectories[genre].values())
            all_labels += [self.labels[genre]]*self.num_trajectories[genre]

        return (all_trajectories, all_labels)
    
    def get_all_encoded_features(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns a list of all the encoded features as well as a list all the genre labels.
        """
        N = sum([sum(self.song_lengths[folder].values()) for folder in self.genres])
        all_features = np.zeros((N, 768))
        all_labels = np.zeros(N, dtype=int)

        i = 0
        for genre in self.genres:
            for file in self.encoded_trajectories[genre]:
                all_features[i:i+self.song_lengths[genre][file]] = self.encoded_trajectories[genre][file]
                all_labels[i:i+self.num_trajectories[genre]] = self.labels[genre]
                i += self.song_lengths[genre][file]

        return (all_features, all_labels)
    
    def split_song(self, song : np.ndarray, num_songlets : int, norm : bool) -> np.ndarray:
        """
        Splits a single song that is (num_timesteps, 768) into (num_songlets, 768).
        """
        if self.encoded_trajectories is None:
            raise Exception("Trajectories not encoded")
        
        self.num_songlets = num_songlets
        songlet_size = song.shape[0] // num_songlets
        songlets = []

        for i in range(num_songlets):
            if i == num_songlets - 1:
                songlets.append(song[i*songlet_size:])
            else:
                songlets.append(song[i*songlet_size:(i+1)*songlet_size])

        if norm:
            return normalize([np.sum(s, axis=0) for s in songlets])
        else:
            return np.stack([np.sum(s, axis=0) for s in songlets])
    
    def split_encoded_trajectories(self, num_songlets : int, norm=True) -> None:
        """
        Splits all the songs into (num_songlets, 768) songlets.
        """
        self.num_songlets = num_songlets
        self.songlet_trajectories = dict()
        for genre in self.genres:
            self.songlet_trajectories[genre] = dict()
            for file in self.encoded_trajectories[genre]:
                self.songlet_trajectories[genre][file] = self.split_song(self.encoded_trajectories[genre][file], num_songlets, norm)
    
    def get_songlet_trajectories(self):
        if self.songlet_trajectories is None:
            raise Exception("Songlets not split")
        return self.songlet_trajectories
    
    def get_all_songlet_trajectories(self) -> tuple[list[np.ndarray], list[int]]:
        """
        Returns a list of all the songlet trajectories as well as a list all the genre labels.
        """
        if self.songlet_trajectories is None:
            raise Exception("Songlets not split")
        
        all_trajectories = np.zeros((sum(self.num_trajectories.values()), self.num_songlets, 768))
        all_labels = np.zeros(sum(self.num_trajectories.values()), dtype=int)

        i = 0
        for genre in self.genres:
            for file in self.trajectories[genre]:
                all_trajectories[i] = self.songlet_trajectories[genre][file]
                all_labels[i] = self.labels[genre]
                i += 1
        
        return (all_trajectories, all_labels)



if __name__ == "__main__":
    # test functons
    data_folders = ["90s_hiphop", "90s_rock", "2010s_pop", "classical", "country"]
    boombox = BoomboxProcessor(verbose=True)
    boombox.load_trajectories(data_folders)
    trajectories = boombox.get_all_trajectories()
    hiphop_trajs = boombox.get_genre_trajectories("90s_hiphop")
    features = boombox.get_all_features()

    from train_encoding_model import BoomboxNet
    boombox.load_encoding_model("models/model_50000.pt", BoomboxNet)
    boombox.encode_trajectories()