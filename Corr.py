import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def main():
    # -------------------------------------------------------------------------
    # 1) Load data
    # -------------------------------------------------------------------------
    # z.dat is assumed to be N x 10 (each row is one sample's 10D latent vector)
    z_data = np.loadtxt("z10.dat")  # shape: (N, 10)
    
    # train.dat is assumed to be N x 37:
    #   Column 0: target
    #   Columns 1..36: all original features (36 total)
    # We only need the first 9 features out of these 36,
    # which in Python indexing are columns 1..9 (slice(1, 10)).
    train_data = np.loadtxt("train.dat")  # shape: (N, 37)
    primary_features = train_data[:, 1:10]  # shape: (N, 9)

    # Basic checks
    N_z, dim_z = z_data.shape
    N_train, dim_train = train_data.shape
    N_primary, dim_primary = primary_features.shape

    print(f"Loaded latent vectors: shape = {z_data.shape} (N={N_z}, dims={dim_z})")
    print(f"Loaded training data : shape = {train_data.shape} (N={N_train}, dims={dim_train})")
    print(f"Primary features     : shape = {primary_features.shape} (N={N_primary}, dims={dim_primary})\n")

    # -------------------------------------------------------------------------
    # 2) Compute correlation: each latent dimension vs each primary feature
    # -------------------------------------------------------------------------
    # We'll create a 10 x 9 correlation matrix: 
    #   rows   = latent dimensions (z1..z10)
    #   columns = primary features (F1..F9)
    corr_matrix = np.zeros((dim_z, dim_primary))

    for i in range(dim_z):
        for j in range(dim_primary):
            # Pearson correlation between latent dimension z_i and feature j
            corr_ij = np.corrcoef(z_data[:, i], primary_features[:, j])[0, 1]
            corr_matrix[i, j] = corr_ij

    # -------------------------------------------------------------------------
    # 3) Print numeric correlation results
    # -------------------------------------------------------------------------
    # Create row/column labels for clarity
    latent_dim_labels   = [f"z{i+1}" for i in range(dim_z)]
    primary_feature_lbl = [f"F{j+1}" for j in range(dim_primary)]

    print("Correlation Matrix (rows = latent dims, columns = primary features):\n")
    df_corr = pd.DataFrame(corr_matrix, 
                           index=latent_dim_labels, 
                           columns=primary_feature_lbl)
    print(df_corr.round(3))  # Round to 3 decimals for neatness
    print()

    # -------------------------------------------------------------------------
    # 4) Visualize correlation as a heatmap
    # -------------------------------------------------------------------------
    plt.figure(figsize=(8, 6))
    sns.heatmap(df_corr, annot=True, fmt=".3f", cmap="RdBu_r", 
                vmin=-1, vmax=1, cbar=True)
    plt.title("Correlation: Latent Dimensions vs. Primary Features")
    plt.xlabel("Primary Features")
    plt.ylabel("Latent Dimensions")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

