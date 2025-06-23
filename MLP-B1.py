#%%%%%%%%%%%%%%%%%%%%%%% %%%%%%%%%%%%%%%%%%%%%%%% %%%%%%%%%%%%%%%%%%%
# This script uses all the raw 36 features in a 5-layer MLP
#          Satadeep (2025)
#%%%%%%%%%%%%%%%%%%%%%%% %%%%%%%%%%%%%%%%%%%%%%%% %%%%%%%%%%%%%%%%%%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import torch.optim as optim
import matplotlib.pyplot as plt


# Set seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)

# Hyperparameters
MLP_LEARNING_RATE = 0.00015
MLP_WEIGHT_DECAY = 1e-5
MLP_PATIENCE = 25
MLP_MIN_DELTA = 5e-4
MLP_NUM_EPOCHS = 1000

TEST_SIZE = 0.20
RANDOM_STATE = 99

# Define the MLP model
class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 64)
        self.fc5 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = F.leaky_relu(self.fc4(x))
        x = self.fc5(x).squeeze()
        return x

# Early stopping
class EarlyStopping:
    def __init__(self, patience=20, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

# Load the data
data = np.loadtxt('train.dat')
y = data[:, 0]
X = data[:, 1:37]
#X = np.loadtxt('z10.dat')
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Split the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Train the MLP
model = MLP(X_train.shape[1])
optimizer = optim.Adam(model.parameters(), lr=MLP_LEARNING_RATE, weight_decay=MLP_WEIGHT_DECAY)
criterion = nn.MSELoss()
early_stopping = EarlyStopping(patience=MLP_PATIENCE, min_delta=MLP_MIN_DELTA)

for epoch in range(MLP_NUM_EPOCHS):
    model.train()
    optimizer.zero_grad()
    y_pred_train = model(X_train_tensor)
    loss = criterion(y_pred_train, y_train_tensor)
    loss.backward()
    optimizer.step()

    # Validate for early stopping
    model.eval()
    with torch.no_grad():
        y_pred_test = model(X_test_tensor)
        val_loss = criterion(y_pred_test, y_test_tensor)
        early_stopping(val_loss.item())
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

# Evaluate the model
model.eval()
with torch.no_grad():
    y_pred_train = model(X_train_tensor).numpy()
    y_pred_test = model(X_test_tensor).numpy()

train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)
test_rmse = mean_squared_error(y_test, y_pred_test, squared=False)
train_rmse = mean_squared_error(y_train, y_pred_train, squared=False)

print(train_r2,train_rmse)

# Plot the results
plt.figure(figsize=(8, 6))
plt.scatter(y_train, y_pred_train, label="Train", alpha=0.7)
plt.scatter(y_test, y_pred_test, label="Test", alpha=0.7, color="orange")
plt.plot([min(y), max(y)], [min(y), max(y)], color="black", linestyle="--")
plt.xlabel("True Values (wt%)")
plt.ylabel("Predictions (wt%)")
#plt.title(f"$R^2 = {test_r2:.3f}$")
plt.legend(loc="best", title=f"Test $R^2$: {test_r2:.2f}\nTest RMSE: {test_rmse:.2f}")
plt.tight_layout()
plt.savefig("MLP_Results.png", dpi=300)
plt.show()

