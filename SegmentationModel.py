import numpy as np
from BoomboxProcessor import BoomboxProcessor
import torch.nn as nn
import torch
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder
from torchmetrics.functional import accuracy as torch_acc
from sklearn.metrics import classification_report
import torch.optim as optim
from torch.utils.data import DataLoader
from train_encoding_model import BoomboxNet

class SegmentationModel:
    def __init__(self) -> None:
        self.bx = BoomboxProcessor()
        self.SKIP_DATA = ['61', '96', '292', '378', '102', '259', '724', '714', '679', '872', '263', '66', '1651', '226', '710', '560', '717', '157', '256', '300', '67', '301', '174', '419', '69', '859', '804', '375', '246', '101', '716', '182']
        self.LABEL_CATS = ['break','bridge','chorus','coda','d/b','development','dialog','end','fade-out','groove','head','instrumental','interlude','intro','main_theme','out','outro','pick-up','post-chorus','pre-chorus','pre-verse','recap','secondary_theme','silence','solo','theme','third','transition','variation','variation_1','variation_2','verse','w/dialog']
    
    def load_data(self) -> None:
        data_folders = ['salami']
        
        self.bx.load_trajectories(data_folders)
        self.bx.load_encoding_model("models/model_50000.pt", BoomboxNet)
        self.bx.encode_trajectories(device='cpu')
        self.bx.split_encoded_trajectories(10,True)
        
        self.songlet_trajectories = self.bx.get_songlet_trajectories()['salami']

        # rename songlet trajectories keys to salami ids
        self.songlet_trajectories = {k.split('/')[-1].split('.')[0]:v for k,v in self.songlet_trajectories.items()}
        self.songlet_trajectories = {k:v for k,v in self.songlet_trajectories.items() if k not in self.SKIP_DATA}

        self.labels = np.load(f"data/salami_labels.npy", allow_pickle=True).item()
        self.labels = {k:v for k,v in self.labels.items() if k not in self.SKIP_DATA}
        

    def create_train_test_data(self) -> None:
        # create train and test data
        self.train_data = np.zeros((len(self.songlet_trajectories), 10, 768))
        self.test_data = np.zeros((len(self.songlet_trajectories), 10, 33))
        
        for i, key in enumerate(self.songlet_trajectories.keys()):
            self.train_data[i] = self.songlet_trajectories[key]
            for k in range(10):
                str_labels = self.labels[key][k]
                # convert str_labels to indecies that they appear in LABEL_CATS
                int_labels = [self.LABEL_CATS.index(str_label) for str_label in str_labels]
                for j in int_labels:
                    self.test_data[i,k,j] = 1
            
                