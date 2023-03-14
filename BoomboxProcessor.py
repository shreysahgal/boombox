import numpy as np
from tqdm import tqdm
import torch

class BoomboxProcessor:
    def __init__(self, verbose:bool=False) -> None:
        self.verbose : bool = verbose

    def load_trajectories(self, data_folders: list[str]):
        """
        Loads the trajectories from the given data folders.
        """
        self.data_folders : list[str] = data_folders
        self.trajectories : dict[str, dict[str, np.ndarray]]= dict()
        self.num_trajectories : dict[str, int] = dict()
        self.song_lengths = dict()

        for folder in self.data_folders:
            self.trajectories[folder] = np.load(f"data/{folder}_trajectories.npy", allow_pickle=True).item()
            self.num_trajectories[folder] = len(self.trajectories[folder])
            self.song_lengths[folder] = {song: trajectory.shape[0] for song, trajectory in self.trajectories[folder].items()}
            if self.verbose:
                print(f"Loaded {self.num_trajectories[folder]} trajectories from {folder}")
        
        self.labels = dict()
        for idx, folder in enumerate(self.data_folders):
            self.labels[folder] = idx


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
        if genre in self.data_folders:
            return (list(self.trajectories[genre].values()), [self.labels[genre]]*self.num_trajectories[genre])
        else:
            raise Exception("Genre not found")
    
    def get_all_trajectories(self) -> tuple[list[np.ndarray], list[int]]:
        """
        Returns a list of all the trajectories as well as a list all the genre labels.
        """
        all_trajectories = []
        all_labels = []
        for genre in self.data_folders:
            all_trajectories += (self.trajectories[genre])
            all_labels += [self.labels[genre]]*self.num_trajectories[genre]

        return (all_trajectories, all_labels)

    def get_all_features(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns a list of all the features as well as a list all the genre labels.
        """
        N = sum([sum(self.song_lengths[folder].values()) for folder in self.data_folders])
        all_features = np.zeros((N, 13, 768))
        all_labels = np.zeros(N, dtype=int)

        i = 0
        for genre in self.data_folders:
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
        for genre in self.data_folders:
            self.encoded_trajectories[genre] = dict()
            iterator = tqdm(self.trajectories[genre]) if self.verbose else self.trajectories[genre]
            for file in iterator:
                new_traj = self.encoding_model.fc1(
                    torch.tensor(self.trajectories[genre][file]).flatten(start_dim=1).float()
                ).detach().numpy()
                self.encoded_trajectories[genre][file] = new_traj.reshape(self.trajectories[genre][file].shape[0], 768)


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