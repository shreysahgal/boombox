from BoomboxProcessor import BoomboxProcessor
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from train_encoding_model import BoomboxNet
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler, random_split
import matplotlib.pyplot as plt
import os
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from functools import partial

class GenreClassifier(nn.Module):
    def __init__(self, l1=1024, l2=128):
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
            nn.AvgPool1d(kernel_size=2, stride=2),
        )
        self.flatten = nn.Flatten()
        self.droput = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(3072, l1)
        self.fc2 = nn.Linear(l1, l2)
        self.fc3 = nn.Linear(l2, 10)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.conv_group(x)
        x = self.flatten(x)
        x = self.droput(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
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
        return self.trajs[idx], self.labels[idx]

def train_epoch(model, device, trainset, loss_fxn, optimizer):
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

def train_gtzan(config, trainset, checkpoint_dir=None):
    model = GenreClassifier(config["l1"], config["l2"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint")
        )
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
    
    trainsize = int(0.8 * len(trainset))
    trainset, valset = random_split(
        trainset, [trainsize, len(trainset) - trainsize]
    )

    trainloader = DataLoader(trainset, batch_size=config["batch_size"], shuffle=True)
    valloader = DataLoader(valset, batch_size=config["batch_size"], shuffle=True)

    train_loss, train_acc, val_loss, val_acc = [], [], [], []\
    
    for epoch in range(config["epochs"]):
        running_loss = 0.0
        epoch_steps = 0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.float().to(device), labels.float().to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, torch.argmax(labels, dim=1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            epoch_steps += 1
            if i % 100 == 99:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / epoch_steps))
                running_loss = 0.0
                epoch_steps = 0
        train_loss.append(running_loss / epoch_steps)

        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        for i, data in enumerate(valloader, 0):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.float().to(device), labels.float().to(device)

                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == torch.argmax(labels, dim=1)).sum().item()
                loss = criterion(outputs, torch.argmax(labels, dim=1))
                val_loss += loss.cpu().numpy()
                val_steps += 1
        
        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((model.state_dict(), optimizer.state_dict()), path)
        
        tune.report(loss=(val_loss / val_steps), accuracy=(correct / total))
    
def test_accuracy(model, test_loader, device):
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = input.float().to(device), labels.float().to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == torch.argmax(labels, dim=1)).sum().item()
    
    return correct / total



def main(num_samples=10, max_num_epochs=10):
    # test model on a single datapoint
    model = GenreClassifier()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    X = torch.randn(1, 5, 768).to(device)
    
    # load the data
    data_folder = "/home/shrey/Documents/eecs448-boombox/data/gtzan/"
    genres = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]

    boombox = BoomboxProcessor(verbose=True)
    boombox.load_trajectories(data_folder, genres, "trajs.pkl")
    boombox.load_encoding_model("models/model_50000.pt", BoomboxNet)
    boombox.encode_trajectories()
    boombox.split_encoded_trajectories(5)
    trajs, labels = boombox.get_all_songlet_trajectories()

    # one-hot encode the labels
    enc = OneHotEncoder(handle_unknown='ignore')
    labels = enc.fit_transform(labels.reshape(-1, 1)).toarray()

    # split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(trajs, labels, test_size=0.2, random_state=42)
    trainset = TrajectoryDataset(X_train, y_train, boombox.num_songlets)
    testset = TrajectoryDataset(X_test, y_test, boombox.num_songlets)

    # set up config and scheduler
    config = {
        "l1": tune.sample_from(lambda _: 2**np.random.randint(6, 11)),
        "l2": tune.sample_from(lambda _: 2**np.random.randint(4, 9)),
        "lr": tune.loguniform(1e-4, 1e-1),
        "weight_decay": tune.loguniform(1e-6, 1e-2),
        "batch_size": tune.choice([32, 64, 128, 256]),
        "epochs": 200
    }

    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=100,
        grace_period=1,
        reduction_factor=2
    )

    reporter = CLIReporter(
        parameter_columns=["l1", "l2", "lr", "weight_decay", "batch_size"],
        metric_columns=["loss", "accuracy", "training_iteration"]
    )

    result = tune.run(
        partial(train_gtzan, trainset=trainset),
        resources_per_trial={"cpu": 1, "gpu": 1},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter
    )

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(best_trial.last_result["accuracy"]))

    # best_model = GenreClassifier(best_trial.config["l1"], best_trial.config["l2"])
    # best_model.to(device)
    # model_state, optimizer_state = torch.load(os.path.join(best_trial.checkpoint.value, "checkpoint"))
    # best_model.load_state_dict(model_state)

    # test_acc = test_accuracy(best_model, testset, device)
    # print("Best trial test set accuracy: {}".format(test_acc))

if __name__ == "__main__":
    main()