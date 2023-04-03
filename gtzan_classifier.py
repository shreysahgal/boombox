from BoomboxProcessor import BoomboxProcessor
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from train_encoding_model import BoomboxNet
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
import matplotlib.pyplot as plt

class GenreClassifier2D(nn.Module):
    def __init__(self):
        super(GenreClassifier2D, self).__init__()
        self.conv_group = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(1,3), stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(1,3), stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

        self.flatten = nn.Flatten()
        self.droput = nn.Dropout(p=0.3)

        self.fc_group = nn.Sequential(
            nn.Linear(3072, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(1024, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = x.view(x.shape[0], 1, x.shape[1], x.shape[2])
        x = self.conv_group(x)
        x = self.flatten(x)
        x = self.droput(x)
        x = self.fc_group(x)
        # print(x.shape)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        # x = self.softmax(x)
        return x




class GenreClassifier(nn.Module):
    def __init__(self):
        super(GenreClassifier, self).__init__()
        
        # self.conv_group = nn.Sequential(
        #     nn.Conv2d(in_channels=5, out_channels=8, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(8),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2)
        # )

        self.conv_group = nn.Sequential(
            nn.Conv1d(in_channels=5, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2, stride=2)
        )

        self.flatten = nn.Flatten()
        self.droput = nn.Dropout(p=0.5)

        self.fc_group = nn.Sequential(
            nn.Linear(16*768, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 10),
            nn.Softmax(dim=1)
        )
        # self.flatten = nn.Flatten()
        # self.droput = nn.Dropout(p=0.5)
        # self.fc1 = nn.Linear(3072, 1024)
        # self.fc2 = nn.Linear(1024, 128)
        # self.fc3 = nn.Linear(128, 10)
        # self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.conv_group(x)
        x = self.flatten(x)
        x = self.droput(x)
        # print(x.shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.softmax(x)
        return x

class LinearGenreClf(nn.Module):
    def __init__(self):
        super(LinearGenreClf, self).__init__()
        self.fc1 = nn.Linear(5*768, 1920)  # 3840
        self.fc2 = nn.Linear(1920, 960)
        self.fc3 = nn.Linear(960, 480)
        self.fc4 = nn.Linear(480, 240)
        self.fc5 = nn.Linear(240, 120)
        self.fc6 = nn.Linear(120, 60)
        self.fc7 = nn.Linear(60, 30)
        self.fc8 = nn.Linear(30, 10)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = self.fc8(x)
        x = self.softmax(x)
        return x

class TrajectoryDataset(Dataset):
    def __init__(self, trajs, labels, S):
        super(Dataset, self).__init__()
        self.trajs = trajs
        self.labels = labels
        self.S = S
    
    def __len__(self):
        return self.trajs.shape[0]

    def __getitem__(self, idx):
        return self.trajs[idx].view(1, 5, 768), self.labels[idx]

def train_epoch(model, device, dataloader, loss_fxn, optimizer):
    model.train()

    train_loss, train_correct = 0.0, 0

    for X, y in dataloader:
        X, y = X.float().to(device), y.float().to(device)
        optimizer.zero_grad()
        output = model(X)
        loss = loss_fxn(output, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * X.size(0)
        _, preds = torch.max(output, 1)
        train_correct += torch.sum(preds == torch.argmax(y, dim=1)).item()
    
    return train_loss, train_correct

def valid_epoch(model, device, dataloader, loss_fxn):
    val_loss, val_correct = 0.0, 0
    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.float().to(device), y.float().to(device)
            output = model(X)
            loss = loss_fxn(output, y)
            val_loss += loss.item() * X.size(0)
            _, preds = torch.max(output, 1)
            val_correct += torch.sum(preds == torch.argmax(y, dim=1)).item()
    
    return val_loss, val_correct

if __name__ == '__main__':
    # test model on a single datapoint
    model = GenreClassifier2D()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    X = torch.randn(1, 5, 768).to(device)
    print(model(X).shape)

    breakpoint()
    
    # load the data
    data_folder = "/home/shrey/Documents/eecs448-boombox/data/gtzan/"
    genres = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]

    boombox = BoomboxProcessor(verbose=True)
    boombox.load_trajectories(data_folder, genres, "trajs.pkl")
    boombox.load_encoding_model("models/model_50000.pt", BoomboxNet)
    boombox.encode_trajectories()
    boombox.split_encoded_trajectories(5)
    trajs, labels = boombox.get_all_songlet_trajectories()

    # trajs = trajs.reshape(trajs.shape[0], trajs.shape[1], trajs.shape[2])

    # one-hot encode the labels
    enc = OneHotEncoder(handle_unknown='ignore')
    labels = enc.fit_transform(labels.reshape(-1, 1)).toarray()

    # split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(trajs, labels, test_size=0.2, random_state=42)

    # create the dataset and k-fold splits
    dataset = TrajectoryDataset(X_train, y_train, boombox.num_songlets)

    # breakpoint()

    n_splits = 5
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # train the model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    train_loss_list = dict()
    test_loss_list = dict()
    train_acc_list = dict()
    test_acc_list = dict()
    iters = dict()
    num_epochs = 100

    for i, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        print("Training fold {}...".format(i+1))
        train_loss_list[i] = []
        test_loss_list[i] = []
        train_acc_list[i] = []
        test_acc_list[i] = []
        iters[i] = []

        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)

        train_loader = DataLoader(dataset, batch_size=32, sampler=train_sampler)
        val_loader = DataLoader(dataset, batch_size=32, sampler=val_sampler)

        model = GenreClassifier2D().to(device)
        loss_fxn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)

        print(f"Fold {i}/{n_splits} training with {len(train_loader.sampler)} training samples and {len(val_loader.sampler)} validation samples...")

        for epoch in range(num_epochs):
            train_loss, train_correct = train_epoch(model, device, train_loader, loss_fxn, optimizer)

            if epoch % 10 == 0:
                val_loss, val_correct = valid_epoch(model, device, val_loader, loss_fxn)

                train_loss_list[i].append(train_loss/len(train_loader.sampler))
                test_loss_list[i].append(val_loss/len(val_loader.sampler))
                train_acc_list[i].append(train_correct/len(train_loader.sampler))
                test_acc_list[i].append(val_correct/len(val_loader.sampler))
                iters[i].append(epoch)

            if epoch % 50 == 0:
                print("Epoch: {}, Train Loss: {}, Train Acc: {}, Val Loss: {}, Val Acc: {}".format(epoch, train_loss/len(train_loader.sampler), train_correct/len(train_loader.sampler), val_loss/len(val_loader.sampler), val_correct/len(val_loader.sampler)))

    # print summary of final train/val accuracy for all folds wity f-strings
    print("------------- Training Summary -------------")
    for i in range(n_splits):
        print(f"Fold {i+1}: Train Acc: {train_acc_list[i][-1]}, Val Acc: {test_acc_list[i][-1]}")
    print("---------------------------------------------")

    # save graphs
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))

    ax[0, 0].set_title("Training Loss")
    ax[0, 1].set_title("Validation Loss")
    ax[1, 0].set_title("Training Accuracy")
    ax[1, 1].set_title("Validation Accuracy")

    ax[1, 0].set_ylim(0, 1)
    ax[1, 1].set_ylim(0, 1)
                    
    for i in range(n_splits):
        ax[0, 0].plot(iters[i], train_loss_list[i], label="Fold {}".format(i+1))
        ax[0, 1].plot(iters[i], test_loss_list[i], label="Fold {}".format(i+1))
        ax[1, 0].plot(iters[i], train_acc_list[i], label="Fold {}".format(i+1))
        ax[1, 1].plot(iters[i], test_acc_list[i], label="Fold {}".format(i+1))

    ax[0, 0].legend()
    ax[0, 1].legend()
    ax[1, 0].legend()
    ax[1, 1].legend()

    fig.savefig("results.png")

    breakpoint()