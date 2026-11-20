import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import mplcursors

learning_rate = 0.001
batch = 32
epochs = 50

class CaliforniaHousingDataset(Dataset):
    def __init__(self, variables, targets):
        self.variables = torch.tensor(variables, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.variables[idx], self.targets[idx]

class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.layer1 = nn.Linear(8, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x

def TrainModel(model, optimizer, criterion, train_loader, test_loader):
    trainLosses, testLosses = [], []
    for epoch in range(epochs):
        model.train()
        trainLoss = 0

        for variables, targets in train_loader:
            outputs = model(variables)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            trainLoss += loss.item()

        trainLoss /= len(train_loader)
        trainLosses.append(trainLoss)

        model.eval()
        testLoss = 0

        with torch.no_grad():
            for variables, targets in test_loader:
                outputs = model(variables)
                loss = criterion(outputs, targets)
                testLoss += loss.item()

        testLoss /= len(test_loader)
        testLosses.append(testLoss)

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {trainLoss:.4f}, Test Loss: {testLoss:.4f}")

    return trainLosses, testLosses

def Visualization(trainLossesDict, testLossesDict, predictionsDict, actuals):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    optimizers = trainLossesDict.keys()

    for idx, optimizer in enumerate(optimizers):
        axes[0, idx].plot(trainLossesDict[optimizer], label="Train Loss")
        axes[0, idx].plot(testLossesDict[optimizer], label="Test Loss")
        axes[0, idx].set_title(f"Loss ({optimizer})")
        axes[0, idx].set_xlabel("Epochs")
        axes[0, idx].set_ylabel("Loss")
        axes[0, idx].legend()

    for idx, optimizer in enumerate(optimizers):
        scatter = axes[1, idx].scatter(actuals, predictionsDict[optimizer], alpha=0.5)
        axes[1, idx].set_title(f"Predicted vs Actual ({optimizer})")
        axes[1, idx].set_xlabel("Actual Prices")
        axes[1, idx].set_ylabel("Predicted Prices")

        cursor = mplcursors.cursor(scatter, hover=True)
        cursor.connect("add", lambda sel: sel.annotation.set_text(f"Actual: {sel.target[0]:.2f}\nPredicted: {sel.target[1]:.2f}"),)

    plt.tight_layout()
    plt.show()

def main():
    data = fetch_california_housing(as_frame=True)
    X = data['data']
    y = data['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)

    y_train = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
    y_test = scaler_y.transform(y_test.values.reshape(-1, 1))

    trainDataset = CaliforniaHousingDataset(X_train, y_train)
    testDataset = CaliforniaHousingDataset(X_test, y_test)

    trainLoader = DataLoader(trainDataset, batch_size=batch, shuffle=True)
    testLoader = DataLoader(testDataset, batch_size=batch, shuffle=False)

    optimizers = {
        "Adam": lambda params: optim.Adam(params, lr=learning_rate),
        "SGD": lambda params: optim.SGD(params, lr=learning_rate),
        "SGD_Momentum": lambda params: optim.SGD(params, lr=learning_rate, momentum=0.9),
    }
    trainLossesDict, testLossesDict, predictionsDict = {}, {}, {}

    for optimizerName, optimizerFn in optimizers.items():
        print(f"\nTraining with {optimizerName}...")

        model = SimpleMLP()
        criterion = nn.MSELoss()
        optimizer = optimizerFn(model.parameters())

        trainLosses, testLosses = TrainModel(model, optimizer, criterion, trainLoader, testLoader)
        trainLossesDict[optimizerName] = trainLosses
        testLossesDict[optimizerName] = testLosses

        model.eval()
        predictions = []
        actuals = []

        with torch.no_grad():
            for features, targets in testLoader:
                outputs = model(features)
                predictions.extend(outputs.numpy())
                actuals.extend(targets.numpy())

        predictions = scaler_y.inverse_transform(predictions)
        actuals = scaler_y.inverse_transform(actuals)
        predictionsDict[optimizerName] = predictions

    Visualization(trainLossesDict, testLossesDict, predictionsDict, actuals)

if __name__ == "__main__":
    main()
