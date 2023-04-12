import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import accuracy as torch_acc

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from BoomboxProcessor import BoomboxProcessor

class BoomboxNet(nn.Module):
    def __init__(self):
      super(BoomboxNet, self).__init__()

      # First fully connected layer
      self.fc1 = nn.Linear(13 * 768, 768)
      
      # Apply pooling to not overfit
      self.p1 = nn.MaxPool1d(4)

      # gradual downsizing
      self.fc2 = nn.Linear(192, 96)

      # normalizing
      self.sm = nn.Softmax()

      # Get outputs
      self.fc3 = nn.Linear(96,5)
      
    def forward(self, x):
      outputs = F.relu(self.fc1(x))
      outputs = F.sigmoid(self.fc2(outputs))
      return outputs

def train(X_train, y_train, X_test, y_test, epochs=100, device="cpu"):
    model = BoomboxNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)  #! could try ADAM here

    losses_train = []
    losses_test = []
    accuracies_test = []
    accuracies_train = []
    iterations = []
    iter = 0

    for epoch in tqdm(range(int(epochs)),desc='Training Epochs'):
        optimizer.zero_grad() # Setting our stored gradients equal to zero
        outputs = model(X_train)
        loss = criterion(torch.squeeze(outputs), y_train) # [200,1] -squeeze-> [200]
        
        loss.backward() # Computes the gradient of the given tensor w.r.t. graph leaves 
        
        optimizer.step() # Updates weights and biases with the optimizer (SGD)
        
        iter+=1

        if iter%1000==0:
            with torch.no_grad():
                # Calculating the loss and accuracy for the test dataset
                outputs_test = torch.squeeze(model(X_test))
                loss_test = criterion(outputs_test, y_test)
                
                predicted_test = torch.zeros_like(outputs_test)
                predicted_test.scatter_(1, torch.argmax(outputs_test, 1).unsqueeze(1), 1)
                accuracy_test = torch_acc(predicted_test, y_test, task='multiclass', num_classes=5).cpu()

                losses_test.append(loss_test.item())
                accuracies_test.append(accuracy_test)
                
                # Calculating the loss and accuracy for the train dataset
                predicted_train = torch.zeros_like(outputs)
                predicted_train.scatter_(1, torch.argmax(outputs, 1).unsqueeze(1), 1)
                accuracy_train = torch_acc(predicted_train, y_train, task='multiclass', num_classes=5).cpu()

                losses_train.append(loss.item())
                accuracies_train.append(accuracy_train)

                iterations.append(iter)

                print(f"Iteration: {iter}. \nTest - Loss: {loss_test.item()}. Accuracy: {accuracy_test}")
                print(f"Train -  Loss: {loss.item()}. Accuracy: {accuracy_train}\n")

                if iter % 10000 == 0:
                    torch.save(model.state_dict(), f"models/model_{iter}.pt")
        
    return losses_train, losses_test, accuracies_train, accuracies_test, iterations

if __name__ == '__main__':
    # load trajectories
    data_folders = ["90s_hiphop", "90s_rock", "2010s_pop", "classical", "country"]
    boombox = BoomboxProcessor()
    boombox.load_trajectories(data_folders)

    # load data and one-hot encode labels
    X, y = boombox.get_all_features()
    enc = OneHotEncoder(handle_unknown='ignore', categories='auto')
    y = enc.fit_transform(y.reshape(-1, 1)).toarray()

    # split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # convert to tensors
    X_train = torch.tensor(X_train).float()
    X_test = torch.tensor(X_test).float()
    y_train = torch.tensor(y_train).float()
    y_test = torch.tensor(y_test).float()

    # train model
    losses_train, losses_test, accuracies_train, accuracies_test, iterations = train(BoomboxNet(), X_train, y_train, X_test, y_test, epochs=100000, device="cuda")

    # plot results
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    axs[0].plot(iterations, losses_train, label="Train")
    axs[0].plot(iterations, losses_test, label="Test")
    axs[0].set_title("Loss")
    axs[0].legend()
    axs[1].plot(iterations, accuracies_train, label="Train")
    axs[1].plot(iterations, accuracies_test, label="Test")
    axs[1].set_title("Accuracy")
    axs[1].legend()
    plt.show()
