import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

# -------------------------------------------------
# Set seeds for reproducibility
# -------------------------------------------------
torch.manual_seed(0)
np.random.seed(0)

# -------------------------------------------------
# Hyperparameters
# -------------------------------------------------
AE_LEARNING_RATE = 0.00015
AE_WEIGHT_DECAY  = 1e-5
AE_PATIENCE      = 25
AE_MIN_DELTA     = 5e-4
AE_NUM_EPOCHS    = 500

MLP_LEARNING_RATE   = 0.00015
MLP_WEIGHT_DECAY    = 1e-5
MLP_PATIENCE        = 25
MLP_MIN_DELTA       = 5e-4
MLP_NUM_EPOCHS      = 1000

TEST_SIZE     = 0.20
RANDOM_STATE  = 99
LATENT_DIM    = 10  # <-- Hard-coded latent dimension

# -------------------------------------------------
# Model Definitions
# -------------------------------------------------
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
        x_recon = self.decoder(z)
        return x_recon

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

# -------------------------------------------------
# EarlyStopping Helper
# -------------------------------------------------
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

# -------------------------------------------------
# Training Functions
# -------------------------------------------------
def train_autoencoder(model, optimizer, criterion, train_data, num_epochs, early_stopping):
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        reconstructed = model(train_data)
        loss = criterion(reconstructed, train_data)
        loss.backward()
        optimizer.step()

        # Early stopping
        early_stopping(loss.item())
        if early_stopping.early_stop:
            print(f"Early stopping triggered for Autoencoder at epoch {epoch+1}")
            break

def train_mlp(model, optimizer, criterion,
              train_features_tensor, train_targets_tensor,
              test_features_tensor, test_targets_tensor,
              num_epochs, early_stopping):
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
    """Compute RMSE and R^2 for the given model and data."""
    model.eval()
    with torch.no_grad():
        predictions = model(features_tensor)
        loss = criterion(predictions, torch.from_numpy(targets_numpy).float())
        rmse = torch.sqrt(loss)
        r2 = r2_score(targets_numpy, predictions.numpy())
    return rmse.item(), r2

# -------------------------------------------------
# Main Script
# -------------------------------------------------
if __name__ == "__main__":
    # 1) Load training data
    input_data = np.loadtxt('train.dat')
    y_full = input_data[:, 0]            # Targets
    x_full = input_data[:, 1:37]         # 36 features
    x_full = (x_full - np.mean(x_full)) / np.std(x_full)  # original normalization


    # 3) Train Autoencoder
    # -----------------------------------
    train_data_tensor = torch.tensor(x_full, dtype=torch.float32)
    input_dim = train_data_tensor.shape[1]

    model_ae = Autoencoder(input_dim, LATENT_DIM)
    optimizer_ae = optim.Adam(model_ae.parameters(), lr=AE_LEARNING_RATE, weight_decay=AE_WEIGHT_DECAY)
    criterion_ae = nn.MSELoss()
    early_stopping_ae = EarlyStopping(patience=AE_PATIENCE, min_delta=AE_MIN_DELTA)

    print(f"Training Autoencoder with latent_dim = {LATENT_DIM} ...")
    train_autoencoder(model_ae, optimizer_ae, criterion_ae,
                      train_data_tensor, AE_NUM_EPOCHS, early_stopping_ae)

    # 4) Extract latent representations (z) for the entire dataset
    with torch.no_grad():
        z_full = model_ae.encoder(train_data_tensor).numpy()

    # 5) Train an MLP on these latent features
    #    a) Split into train/test sets
    train_features, test_features, train_targets, test_targets = train_test_split(
        z_full, y_full, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    #    b) Scale the latent space for MLP
    z_scaler = StandardScaler()
    train_features_scaled = z_scaler.fit_transform(train_features)
    test_features_scaled  = z_scaler.transform(test_features)

    train_features_tensor = torch.from_numpy(train_features_scaled).float()
    test_features_tensor  = torch.from_numpy(test_features_scaled).float()
    train_targets_tensor  = torch.from_numpy(train_targets).float()
    test_targets_tensor   = torch.from_numpy(test_targets).float()

    #    c) Define and train MLP
    model_mlp = MLP(LATENT_DIM)
    optimizer_mlp = optim.Adam(model_mlp.parameters(), lr=MLP_LEARNING_RATE, weight_decay=MLP_WEIGHT_DECAY)
    criterion_mlp = nn.MSELoss()
    early_stopping_mlp = EarlyStopping(patience=MLP_PATIENCE, min_delta=MLP_MIN_DELTA)

    print("Training MLP on latent features ...")
    train_mlp(model_mlp, optimizer_mlp, criterion_mlp,
              train_features_tensor, train_targets_tensor,
              test_features_tensor, test_targets_tensor,
              MLP_NUM_EPOCHS, early_stopping_mlp)

    # 6) Evaluate and plot
    # -----------------------------------
    train_rmse, train_r2 = evaluate_model(model_mlp, train_features_tensor, train_targets, criterion_mlp)
    test_rmse,  test_r2  = evaluate_model(model_mlp,  test_features_tensor,  test_targets,  criterion_mlp)

    # Predictions for scatter plot
    model_mlp.eval()
    with torch.no_grad():
        train_preds = model_mlp(train_features_tensor).numpy()
        test_preds  = model_mlp(test_features_tensor).numpy()

    # Scatter plot
    plt.figure(figsize=(6, 6))
    plt.scatter(train_targets, train_preds, label="Train", alpha=0.8)
    plt.scatter(test_targets,  test_preds,  label="Test",  alpha=0.8)

    # Plot diagonal
    all_vals = np.concatenate([train_targets, test_targets, train_preds, test_preds])
    min_val, max_val = 0.0, max(all_vals) * 1.1
    plt.plot([min_val, max_val], [min_val, max_val], 'k--')

    plt.xlim([min_val, max_val])
    plt.ylim([min_val, max_val])
    plt.xlabel("True Values")
    plt.ylabel("Predictions")
    plt.legend()

    plt.text(
        0.05 * max_val, 0.90 * max_val,
        f"Test R$^2$: {test_r2:.2f}\nTest RMSE: {test_rmse:.2f}"
    )
    plt.title(f"AE+MLP (latent_dim={LATENT_DIM})\nTrain R²={train_r2:.2f}, Test R²={test_r2:.2f}")
    plt.tight_layout()
    plt.savefig("AE_MLP_scatter.png", dpi=300)
    plt.show()

    print(f"Final Results (latent_dim={LATENT_DIM}):")
    print(f"  Train RMSE: {train_rmse:.2f}, Train R²: {train_r2:.2f}")
    print(f"  Test  RMSE: {test_rmse:.2f}, Test  R²: {test_r2:.2f}")

    # 7) (Optional) Save models if desired
    # torch.save(model_ae.state_dict(),  "autoencoder_dim10.pth")
    # torch.save(model_mlp.state_dict(), "mlp_dim10.pth")
    # You can also pickle your scalers if you want to reload later.

    # --------------------------------------------------------
    # 8) Predict on unknown.dat (containing 36 features only)
    # --------------------------------------------------------
    # Suppose unknown.dat has rows of 36 features each (no target).
    try:
        unknown_features = np.loadtxt('unknown.dat', usecols=range(1,37))
    except OSError:
        print("unknown.dat not found. Skipping prediction on unknown data.")
        exit()

    # If unknown.dat has only one line, ensure it's shape=(1,36)
    if unknown_features.ndim == 1:
        unknown_features = unknown_features.reshape(1, -1)

    # Same normalization as training data
    unknown_features_norm = (unknown_features - np.mean(unknown_features)) / np.std(unknown_features)
    unknown_tensor = torch.from_numpy(unknown_features_norm).float()

    # Encode into latent space
    with torch.no_grad():
        z_unknown = model_ae.encoder(unknown_tensor).numpy()

    # Scale latent space as for MLP
    z_unknown_scaled = z_scaler.fit_transform(z_unknown)
    z_unknown_tensor = torch.from_numpy(z_unknown_scaled).float()

    # Predict
    with torch.no_grad():
        unknown_predictions = model_mlp(z_unknown_tensor).numpy()

    # Print or save predictions
    print("\nPredictions for unknown materials:")
    print(unknown_predictions)

