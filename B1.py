#This script transforms the original features to a latent 
# represntation and then feeds to a 5-layer MLP
# Satadeep Bhattacharjee
# IKST, Bangalore
#%%%%%%%%%%%%%%%%%%%%%%% %%%%%%%%%%%%%%%%%%%%%%%% %%%%%%%%%%%%%%%%%%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import torch.optim as optim
import matplotlib.pyplot as plt

# Set seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)

# Hyperparameters (centralized)
AE_LEARNING_RATE = 0.00015
AE_WEIGHT_DECAY = 1e-5
AE_PATIENCE = 25
AE_MIN_DELTA = 5e-4
AE_NUM_EPOCHS = 500

MLP_LEARNING_RATE = 0.00015
MLP_WEIGHT_DECAY = 1e-5
MLP_PATIENCE = 25
MLP_MIN_DELTA = 5e-4
MLP_NUM_EPOCHS = 1000


LATENT_DIMS = list(range(4, 24, 2))  # {4, 6, 8, 10, 12, 14, 16, 18, 20,22,24}
TEST_SIZE = 0.20
RANDOM_STATE = 99

#-------------------------------------
# Model Definitions
#-------------------------------------
class Autoencoder(nn.Module):
    def __init__(self, dim_in, dim_latent):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(dim_in, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, dim_latent)
        )
        self.decoder = nn.Sequential(
            nn.Linear(dim_latent, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, dim_in)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_p = self.decoder(z)
        return x_p

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

#-------------------------------------
# Helper Classes and Functions
#-------------------------------------
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

def train_autoencoder(model, optimizer, criterion, train_data, num_epochs, early_stopping):
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        reconstructed = model(train_data)
        loss = criterion(reconstructed, train_data)
        loss.backward()
        optimizer.step()

        early_stopping(loss.item())
        if early_stopping.early_stop:
            print(f"Early stopping triggered for Autoencoder at epoch {epoch+1}")
            break

def train_mlp(model, optimizer, criterion, train_features_tensor, train_targets_tensor,
              test_features_tensor, test_targets_tensor, num_epochs, early_stopping):
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        train_pred = model(train_features_tensor)
        train_loss = criterion(train_pred, train_targets_tensor)
        train_loss.backward()
        optimizer.step()

        # Validation loss for early stopping
        model.eval()
        with torch.no_grad():
            test_pred = model(test_features_tensor)
            test_loss = criterion(test_pred, test_targets_tensor)

        early_stopping(test_loss.item())
        if early_stopping.early_stop:
            print(f"Early stopping triggered for MLP at epoch {epoch+1}")
            break

def evaluate_model(model, features_tensor, targets_numpy, criterion):
    model.eval()
    with torch.no_grad():
        predictions = model(features_tensor)
        loss = criterion(predictions, torch.from_numpy(targets_numpy).float())
        rmse = torch.sqrt(loss)
        r2 = r2_score(targets_numpy, predictions.numpy())
    return rmse.item(), r2

#-------------------------------------
# Main Execution
#-------------------------------------
# Load the training data once
input_data = np.loadtxt('train.dat')
column_data = input_data[:, 0]
y_full = column_data
x_full = input_data[:, 1:37]
x_full = (x_full - np.mean(x_full)) / np.std(x_full)  # original normalization


test_r2_scores = []
test_rmse_scores = []

for latent_dimension in LATENT_DIMS:
    print(f'\n***** Latent Dimension: {latent_dimension} *****')

    # Train the Autoencoder
    train_data_tensor = torch.tensor(x_full, dtype=torch.float32)
    input_dim = train_data_tensor.shape[1]
    model_ae = Autoencoder(input_dim, latent_dimension)
    criterion_ae = nn.MSELoss()
    optimizer_ae = optim.Adam(model_ae.parameters(), lr=AE_LEARNING_RATE, weight_decay=AE_WEIGHT_DECAY)
    early_stopping_ae = EarlyStopping(patience=AE_PATIENCE, min_delta=AE_MIN_DELTA)

    train_autoencoder(model_ae, optimizer_ae, criterion_ae, train_data_tensor, AE_NUM_EPOCHS, early_stopping_ae)

    # Extract latent representations
    with torch.no_grad():
        z = model_ae.encoder(train_data_tensor).numpy()
    z_file = f'z{latent_dimension}.dat'
#    np.savetxt('z.dat', z, delimiter=' ', fmt='%.6f')
    np.savetxt(z_file, z, delimiter=' ', fmt='%.6f')

    # MLP Regression
    targets = np.loadtxt('train.dat', usecols=[0])
    features = np.loadtxt(z_file)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    num_features = features_scaled.shape[1]
    train_features, test_features, train_targets, test_targets = train_test_split(
        features_scaled, targets, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    train_features_tensor = torch.from_numpy(train_features).float()
    test_features_tensor = torch.from_numpy(test_features).float()
    train_targets_tensor = torch.from_numpy(train_targets).float()
    test_targets_tensor = torch.from_numpy(test_targets).float()

    model_mlp = MLP(num_features)
    optimizer_mlp = optim.Adam(model_mlp.parameters(), lr=MLP_LEARNING_RATE, weight_decay=MLP_WEIGHT_DECAY)
    criterion_mlp = nn.MSELoss()
    early_stopping_mlp = EarlyStopping(patience=MLP_PATIENCE, min_delta=MLP_MIN_DELTA)

    train_mlp(model_mlp, optimizer_mlp, criterion_mlp, train_features_tensor, train_targets_tensor,
              test_features_tensor, test_targets_tensor, MLP_NUM_EPOCHS, early_stopping_mlp)

    # Evaluate model
    train_rmse, r2_train = evaluate_model(model_mlp, train_features_tensor, train_targets, criterion_mlp)
    test_rmse, r2_test = evaluate_model(model_mlp, test_features_tensor, test_targets, criterion_mlp)

    # Store metrics
    test_r2_scores.append(r2_test)
    test_rmse_scores.append(test_rmse)

    # Log progress
    print(
        f"LatentDim={latent_dimension}, Train-RMSE={train_rmse:.3f}, "
        f"Test-RMSE={test_rmse:.3f}, Train-R²={r2_train:.2f}, Test-R²={r2_test:.2f}"
    )

#-------------------------------------
# Plotting bar chart: Latent dimension vs R^2 and RMSE (test)
#-------------------------------------
fig, ax1 = plt.subplots(figsize=(10, 6))  # Adjust figure size for better readability

x_axis = np.arange(len(LATENT_DIMS))
width = 0.35

# Plot R^2 as bars
bars_r2 = ax1.bar(x_axis - width/2, test_r2_scores, width, label='Test R²', color='blue')
ax1.set_xlabel('Latent Dimension')
ax1.set_ylabel('R²', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.set_xticks(x_axis)
ax1.set_xticklabels(LATENT_DIMS)
ax1.set_ylim([0, 1])
ax1.set_yticks(np.arange(0, 1.1, 0.1))

# Create a second y-axis for RMSE
ax2 = ax1.twinx()
bars_rmse = ax2.bar(x_axis + width/2, test_rmse_scores, width, label='Test RMSE', color='orange')
ax2.set_ylabel('RMSE', color='orange')
ax2.tick_params(axis='y', labelcolor='orange')
ax2.set_ylim([0, 1])
ax2.set_yticks(np.arange(0, 1.1, 0.1))

# Add a legend
fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=2)  # More clear legend placement

plt.title('Latent Dimension vs Test R² and RMSE')
plt.tight_layout()
plt.savefig('Latent_vs_R2_RMSE.png', dpi=300)
plt.show()
