import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

def train_epoch(model, train_loader, criterion, optimizer, device='cpu'):
    model.train()
    total_loss = 0
    total, correct = 0, 0
    for i, (x, y) in enumerate(train_loader):
        x = x.to(device).float()
        y = y.to(device).long()
        optimizer.zero_grad()
        output = model(x)
        # breakpoint()
        loss = criterion(output.to(device), y.to(device))
        correct += (torch.argmax(output, dim=1) == y).sum().item()
        total += y.size(0)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader), correct / total

def eval_epoch(model, val_loader, criterion, device='cpu'):
    model.eval()
    total_loss = 0
    total, correct = 0, 0
    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            x = x.to(device).float()
            y = y.to(device).long()
            output = model(x)
            loss = criterion(output, y)
            correct += (torch.argmax(output, dim=1) == y).sum().item()
            total += y.size(0)
            total_loss += loss.item()
    return total_loss / len(val_loader), correct / total

class TrajectoryDataset(Dataset):
    def __init__(self, trajectories, labels, transform=None):
        self.data = trajectories
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        if self.transform:
            x = self.transform(x)
        return x, y
    
class SongletClassifier(nn.Module):
    def __init__(self):
        super(SongletClassifier, self).__init__()
        self.fc_group = nn.Sequential(
            nn.Linear(768, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 14),
            nn.Softmax(dim=1)
        )
    def forward(self, x):
        x = self.fc_group(x)
        return x

def main():
    NUM_EPOCHS = 100
    BATCH_SIZE = 32

    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-6

    NUM_SONGLETS = 10
    NUM_FOLDS = 5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    LOG_EPOCHS = 5
    PLOT_SAVE_PATH = "plots/large_dataset_songlet_clf.png"
    CF_SAVE_PATH = "plots/large_dataset_songlet_clf_cf.png"

    # Load data
    df = pd.read_pickle("data/large_dataset/correct_trajectories.pkl")

    label_dict = {genre: idx for idx, genre in enumerate(df["genre"].unique())}
    print(label_dict)

    X = np.concatenate(df["trajectory"].to_numpy()).reshape(-1, 768)
    y = np.array([label_dict[genre] for genre in df["genre"].to_numpy()])
    y = np.repeat(y, NUM_SONGLETS)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create dataset
    train_dataset = TrajectoryDataset(X_train, y_train)

    train_loss_list = {fold: [] for fold in range(NUM_FOLDS)}
    train_acc_list = {fold: [] for fold in range(NUM_FOLDS)}
    val_loss_list = {fold: [] for fold in range(NUM_FOLDS)}
    val_acc_list = {fold: [] for fold in range(NUM_FOLDS)}
    train_iters = {fold: [] for fold in range(NUM_FOLDS)}
    val_iters = {fold: [] for fold in range(NUM_FOLDS)}
    cf_matrices = {fold: None for fold in range(NUM_FOLDS)}

    recent_val_acc = 0
    recent_val_loss = 0
    
    for fold, (train_idx, val_idx) in enumerate(KFold(n_splits=NUM_FOLDS).split(X_train)):
        print(f"Fold {fold + 1}/{NUM_FOLDS}")
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
        val_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=val_sampler)

        # Create model
        model = SongletClassifier().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

        for epoch in (pbar := tqdm(range(NUM_EPOCHS))):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            if epoch % LOG_EPOCHS == 0:
                recent_val_loss, recent_val_acc = eval_epoch(model, val_loader, criterion, device)
                val_loss_list[fold].append(recent_val_loss)
                val_acc_list[fold].append(recent_val_acc)
                val_iters[fold].append(epoch)
            pbar.set_description(f"Epoch {epoch + 1}/{NUM_EPOCHS} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {recent_val_loss:.4f} | Val Acc: {recent_val_acc:.4f}")

            train_loss_list[fold].append(train_loss)
            train_acc_list[fold].append(train_acc)
            train_iters[fold].append(epoch)
        
        # create confusion matrix with test data
        test_dataset = TrajectoryDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
        test_loss, test_acc = eval_epoch(model, test_loader, criterion, device)
        print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
        y_pred = []
        y_true = []
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                y = y.to(device)
                y_pred.append(model(x).argmax(dim=1))
                y_true.append(y)
        y_pred = torch.cat(y_pred).cpu().numpy()
        y_true = torch.cat(y_true).cpu().numpy()
        cm = confusion_matrix(y_true, y_pred)
        cf_matrices[fold] = cm


            # print(f"Epoch {epoch + 1}/{NUM_EPOCHS} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    # Plot training and validation loss
    plots_fig, ax = plt.subplots(2, 2)
    plots_fig.set_size_inches(12, 8)
    for fold in range(NUM_FOLDS):
        ax[0, 0].plot(train_iters[fold], train_loss_list[fold])
        ax[0, 1].plot(train_iters[fold], train_acc_list[fold])
        ax[1, 0].plot(val_iters[fold], val_loss_list[fold])
        ax[1, 1].plot(val_iters[fold], val_acc_list[fold])
    ax[0, 0].set_title("Training Loss")
    ax[0, 1].set_title("Training Accuracy")
    ax[1, 0].set_title("Validation Loss")
    ax[1, 1].set_title("Validation Accuracy")
    plots_fig.savefig(PLOT_SAVE_PATH)

    # plot confusion matrices
    cf_fig, ax = plt.subplots(1, NUM_FOLDS)
    cf_fig.set_size_inches(12, 4)
    for fold in range(NUM_FOLDS):
        sns.heatmap(cf_matrices[fold], annot=False, ax=ax[fold])
        ax[fold].set_title(f"Fold {fold + 1}")
    cf_fig.tight_layout()
    cf_fig.savefig(CF_SAVE_PATH)
    

    


if __name__ == "__main__":
    main()