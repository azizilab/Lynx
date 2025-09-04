# %%
import os
import subprocess

import anndata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import random

# %% Define the path to the simulate_spatial_data.py script
simulate_spatial_data_script_path = "./simulate_spatial_data_3ct.py"

# Define the path to the interaction CSV and the output path for the generated dataset
interaction_csv_path = "../data/3ct_sweep_interaction_csv.csv"
output_path = f"../data/3ct_sweep_dataset.h5ad"

ct1, ct2, ct3 = "3", "0", "2"
interaction_df = pd.DataFrame(
    [
        {
            "receptor_cell": ct1,
            "receptor_subtype": f"{ct1}_sub0",
            "interaction_type": "neutral",
        },
        {
            "receptor_cell": ct2,
            "receptor_subtype": f"{ct2}_sub0",
            "interaction_type": "neutral",
        },
        {
            "receptor_cell": ct3,
            "receptor_subtype": f"{ct3}_sub0",
            "interaction_type": "neutral",
        },
        {
            "receptor_cell": ct2,
            "sender_cell": ct1,
            "receptor_subtype": f"{ct2}_sub1",
            "radius_of_effect": 20,
            "interaction_type": "interaction",
        },
        {
            "receptor_cell": ct3,
            "sender_cell": ct2,
            "receptor_subtype": f"{ct3}_sub1",
            "radius_of_effect": 10,
            "interaction_type": "interaction",
        },
    ]
)
interaction_df.to_csv(interaction_csv_path, index=False)
# %%
interaction_df

# %%
# Construct the command to run the simulate_spatial_data.py script
# Only do this if the data doesnt already exist
if not os.path.exists(output_path):
    command = f"python3 {simulate_spatial_data_script_path} {interaction_csv_path} {output_path}"
    # Execute the command
    res = subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
    )
    for line in str(res.stdout, encoding="utf-8").splitlines():
        print(line)
    for line in str(res.stderr, encoding="utf-8").splitlines():
        print(line)

    adata = anndata.read_h5ad(output_path)

    adata.write_h5ad(output_path)
else:
    adata = anndata.read_h5ad(output_path)

# %%
# Viz dataset
plt.figure(figsize=(8, 6))
plot_df = adata.obsm["spatial"].copy()
plot_df["subtype"] = adata.obs["subtype"]
sns.scatterplot(plot_df, x="X", y="Y", hue="subtype", alpha=0.7, s=10)

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Spatial Distribution of Cells with Subtype Annotations")
plt.legend()
plt.savefig("../figures/3ct_sweep_dataset_viz.png")
plt.show()

# %%
seed = 42
random.seed(seed)
np.random.seed(seed)

# %%
# split test by spatial location
train_path = f"../data/3ct_sweep_dataset_train.h5ad"
test_path = f"../data/3ct_sweep_dataset_test.h5ad"
if not os.path.exists(train_path) or not os.path.exists(test_path):
    test_indices = np.where(
        (adata.obsm["spatial"].iloc[:, 0] > 900)
        & (adata.obsm["spatial"].iloc[:, 0] < 1100)
    )[0]
    train_indices = np.setdiff1d(np.arange(adata.shape[0]), test_indices)

    adata.obs["train_test_split"] = "unassigned"
    adata.obs.iloc[train_indices, adata.obs.columns.get_loc("train_test_split")] = (
        "train"
    )
    adata.obs.iloc[test_indices, adata.obs.columns.get_loc("train_test_split")] = "test"
    adata.write_h5ad(output_path)

    adata_train = adata[train_indices]
    adata_test = adata[test_indices]
    adata_train.write_h5ad(f"../data/3ct_dataset_train.h5ad")
    adata_test.write_h5ad(f"../data/3ct_dataset_test.h5ad")
else:
    adata_train = anndata.read_h5ad(train_path)
    adata_test = anndata.read_h5ad(test_path)

# %% Viz the train and test datasets for clarity
plt.figure(figsize=(8, 6))
plot_df = adata_train.obsm["spatial"].copy()
plot_df["subtype"] = adata_train.obs["subtype"]
sns.scatterplot(plot_df, x="X", y="Y", hue="subtype", alpha=0.7, s=10)

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Spatial Distribution of Cells with Subtype Annotations (Train)")
plt.legend()
plt.savefig("../figures/3ct_dataset_viz_train.png")
plt.show()

plt.figure(figsize=(8, 6))
plot_df = adata_test.obsm["spatial"].copy()
plot_df["subtype"] = adata_test.obs["subtype"]
sns.scatterplot(plot_df, x="X", y="Y", hue="subtype", alpha=0.7, s=10)

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Spatial Distribution of Cells with Subtype Annotations (Test)")
plt.legend()
plt.savefig("../figures/3ct_dataset_viz_test.png")
plt.show()
# %%
