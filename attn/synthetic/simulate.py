import os
import random

import anndata as ad
import click
import numpy as np
import pandas as pd
import scanpy as sc

from sklearn.mixture import GaussianMixture
from scipy.stats import truncnorm




def generate_synthetic_dataset(adata, n_subclusters=2, random_state=42):
    # random.seed(random_state)
    # np.random.seed(random_state)

    adata = _preprocess(adata, n_hvgs=500, random_state=random_state)
    # adata = _subcluster(
    #     adata, n_subcluster_per_cluster=n_subclusters, n_genes_for_subclustering=50,
    #     random_state=random_state
    # )

    return adata
def generate_coordinates(adata, interaction_df, num_cells=6000, use_confounder=True, confounder_prob=1e-2, confounder_noise=False):
    
    adata = adata.copy()

    #cts[-1] is allways the confounder
    cts = interaction_df.receptor_cell.unique()
    confounder_ct = cts[-1]
    adata.obs.loc[adata.obs['subtype'] == f'{confounder_ct}_sub1', 'subtype'] = f'{confounder_ct}_sub0'
    
    spatial_ad = _generate_2d_triangular_gradient_data(
        adata,
        interaction_df,
        num_cells=num_cells,
        rect_length=2000,
        rect_width=1000,
        use_confounder=use_confounder,
        confounder_prob=confounder_prob,
        confounder_noise=confounder_noise
    )

    #squidpy processing
    spatial_ad.obsm['spatial'] = spatial_ad.obsm['spatial'].values

    sample_id = 'sample'
    img = None  # Placeholder w/ empty entry for `adata.uns`
            
    spatial_ad.uns['spatial'] = {
        sample_id: {
            'images': {'hires': img}, 
            'scalefactors': {
                'spot_diameter_fullres': 1.0, 
                'tissue_hires_scalef': 1.0
            }
        }
    }
    return spatial_ad


def _preprocess(adata, n_hvgs=2000, random_state=42):
    adata_log = adata.copy()
    adata_log.layers["counts"] = adata_log.X.copy()
    sc.pp.normalize_total(adata_log)
    sc.pp.log1p(adata_log)
    sc.pp.highly_variable_genes(
        adata_log, flavor="seurat_v3", layer="counts", n_top_genes=n_hvgs, subset=True
    )
    sc.tl.pca(adata_log)

    sc.pp.neighbors(adata_log)
    sc.tl.leiden(adata_log, resolution=0.3, key_added="leiden", random_state=random_state)

    # Prune small clusters
    cluster_counts = adata_log.obs["leiden"].value_counts()
    clusters_to_keep = cluster_counts[cluster_counts >= 1000].index
    adata_log = adata_log[adata_log.obs["leiden"].isin(clusters_to_keep)]

    # Redo with new clusters
    sc.tl.leiden(adata_log, resolution=0.3, key_added="leiden", random_state=random_state)
    sc.tl.umap(adata_log, random_state=random_state)

    return adata_log

def _subcluster(adata, n_subcluster_per_cluster=3, n_genes_for_subclustering=200, random_state=42):
    print("Number of subclusters per cluster:", n_subcluster_per_cluster)
    rng = np.random.default_rng(random_state)

    adata.obs["subtype"] = "unassigned"

    for i, ct in enumerate(adata.obs["leiden"].unique(), start=1):

        cluster_mask = adata.obs["leiden"] == ct
        gene_avg = np.ravel(adata[cluster_mask].X.mean(axis=0))

        n_top = n_genes_for_subclustering           
        sorted_idx = np.argsort(gene_avg)
        top_idx = sorted_idx[-n_top:]

        # rand_idx = rng.choice(top_idx, n_genes_for_subclustering, replace=False)

        selected_genes = adata.var_names[top_idx]

        print(f"\nCluster {ct}: {len(selected_genes)} genes")

        col = f"is_gene_for_subclustering_{ct}"
        adata.var[col] = False
        adata.var.loc[selected_genes, col] = True

        adata_sub = adata[:, adata.var[col]].copy()
        sc.pp.pca(adata_sub, n_comps=20)
        adata.obsm[f"X_rep_subclustering_{ct}"] = adata_sub.obsm["X_pca"]

        X = adata[cluster_mask].obsm[f"X_rep_subclustering_{ct}"]
        gmm = GaussianMixture(n_components=n_subcluster_per_cluster, random_state=random_state)
        gmm_labels = gmm.fit_predict(X)

        adata.obs.loc[cluster_mask, "subtype"] = [f"{ct}_sub{l}" for l in gmm_labels]

    adata.obs["subtype"] = adata.obs["subtype"].astype("category")
    return adata
# def _subcluster(adata, n_subcluster_per_cluster=3, n_genes_for_subclustering=200):
#     print("Number of subclusters per cluster:", n_subcluster_per_cluster)

#     gene_ratio = 0.5

#     adata.obs["subtype"] = "unassigned"
#     for ct in adata.obs["leiden"].unique():
#         gene_avg_expression = np.ravel(adata.X.mean(axis=0))
#         sorted_genes_by_expression = np.argsort(gene_avg_expression)
#         n_top_genes = int(gene_ratio * n_genes_for_subclustering)
#         n_remaining_genes = n_genes_for_subclustering - n_top_genes
#         top_genes_indices, remaining_genes_indices = (
#             sorted_genes_by_expression[-n_top_genes:],
#             sorted_genes_by_expression[:n_remaining_genes],
#         )
#         selected_genes_for_subclustering = adata.var_names[top_genes_indices]
#         selected_genes_for_subclustering = np.concatenate(
#             [
#                 selected_genes_for_subclustering,
#                 np.random.choice(
#                     adata.var_names[remaining_genes_indices],
#                     n_remaining_genes,
#                     replace=False,
#                 ),
#             ]
#         )
#         adata.var[f"is_gene_for_subclustering_{ct}"] = False
#         adata.var.loc[
#             selected_genes_for_subclustering, f"is_gene_for_subclustering_{ct}"
#         ] = True
#         adata_sub = adata[:, adata.var[f"is_gene_for_subclustering_{ct}"]].copy()

#         sc.pp.pca(adata_sub, n_comps=20)
#         adata.obsm[f"X_rep_subclustering_{ct}"] = adata_sub.obsm["X_pca"]
#         _leiden_subclustering_binary_search(adata, ct, n_subcluster_per_cluster)
#     adata.obs["subtype"] = adata.obs["subtype"].astype("category")
#     return adata


def _generate_2d_triangular_gradient_data(
    adata,
    interaction_df,
    num_cells=6000,
    rect_length=2000,
    rect_width=1000,
    use_confounder=True,
    confounder_prob=1e-2,
    confounder_noise=False
):
    cts = interaction_df.receptor_cell.unique()

    #ct4 is confounder
    ct1, ct2, ct3, ct4 = cts

    # rect_length is the dimension of the gradient
    gradient_width = 700

    # Generate uniform spatial positions
    positions = np.random.uniform(0, (rect_length, rect_width), size=(num_cells, 2))

    # Calculate midpoints for the lines
    mid_x = rect_length / 2
    mid_y = rect_width / 2

    # Calculate gradient zones around the lines
    gradient_x_start = mid_x - gradient_width / 2
    gradient_x_end = mid_x + gradient_width / 2
    gradient_y_start = mid_y - gradient_width / 2
    gradient_y_end = mid_y + gradient_width / 2

    # Assign cell types based on the position and gradient probabilities
    cell_types = []
    for pos in positions:
        x_gradient_prob = (
            np.clip((pos[0] - gradient_x_start) / gradient_width, 0, 1)
            if gradient_x_start <= pos[0] <= gradient_x_end
            else 1
        )
        y_gradient_prob = (
            np.clip((pos[1] - gradient_y_start) / gradient_width, 0, 1)
            if gradient_y_start <= pos[1] <= gradient_y_end
            else 1
        )

        if pos[0] < mid_x:  # Left half of the rectangle
            if pos[1] < mid_y:  # Bottom left quadrant
                cell_types.append(ct1 if np.random.rand() < x_gradient_prob else ct3)
            else:  # Top left quadrant, with gradient to ct2
                cell_types.append(ct2 if np.random.rand() < y_gradient_prob else ct1)
        else:  # Right half of the rectangle
            if pos[1] < mid_y:  # Bottom right quadrant, with gradient to ct2
                cell_types.append(ct3 if np.random.rand() < x_gradient_prob else ct1)
            else:  # Top right quadrant
                cell_types.append(ct2 if np.random.rand() < y_gradient_prob else ct3)
        

        if use_confounder and np.random.rand() < confounder_prob:
            cell_types.pop()
            cell_types.append(ct4)

    #TODO create gradients over full space and apply to new cell type and all cells types


    # Create a DataFrame with cell positions and types
    spatial_data = pd.DataFrame(
        {
            "Cell_ID": range(1, num_cells + 1),
            "X": positions[:, 0],
            "Y": positions[:, 1],
            "Cell_Type": cell_types,
        }
    )

    rule_set = interaction_df[
        interaction_df["interaction_type"] == "interaction"
    ].to_dict(orient="records")
    neutral_types = dict(
        zip(
            interaction_df[interaction_df["interaction_type"] == "neutral"][
                "receptor_cell"
            ],
            interaction_df[interaction_df["interaction_type"] == "neutral"][
                "receptor_subtype"
            ],
        )
    )
    assert ct1 in neutral_types
    assert ct2 in neutral_types
    assert ct3 in neutral_types
    assert ct4 in neutral_types

    spatial_data["Subtype"] = spatial_data["Cell_Type"].map(neutral_types)
    for rule in rule_set:
        receptor_cells = spatial_data[
            (spatial_data["Cell_Type"] == rule["receptor_cell"])
            & (spatial_data["Subtype"] == neutral_types[rule["receptor_cell"]])
        ]
        sender_cells = spatial_data[spatial_data["Cell_Type"] == rule["sender_cell"]]

        for _, receptor_cell in receptor_cells.iterrows():
            distances = np.sqrt(
                (sender_cells["X"] - receptor_cell["X"]) ** 2
                + (sender_cells["Y"] - receptor_cell["Y"]) ** 2
            )
            if (distances <= rule["radius_of_effect"]).any():
                spatial_data.loc[
                    spatial_data["Cell_ID"] == receptor_cell["Cell_ID"], "Subtype"
                ] = rule["receptor_subtype"]
    #TODO add spatial data to adata
    sampled_adatas = []
    for subtype in spatial_data["Subtype"].unique():
        subtype_cells = adata.obs[adata.obs["subtype"] == subtype].index

        num_samples = (spatial_data["Subtype"] == subtype).sum()

        sampled_indices = np.random.choice(subtype_cells, num_samples, replace=True)
        sampled_spatial = spatial_data.loc[
            spatial_data["Subtype"] == subtype, ["X", "Y"]
        ]
        sampled_adata = adata[sampled_indices].copy()
        sampled_spatial.index = sampled_adata.obs_names
        sampled_adata.obsm["spatial"] = sampled_spatial

        #Add confounding noise based on coordinate
        # if subtype == '4_sub0' or confounder_noise:
        #     noise = truncnorm.rvs(
        #     a=0, b=np.inf, loc=sampled_adata.obsm['spatial'].X + sampled_adata.obsm['spatial'].Y, scale=1
        #     )
        #     noise = noise[:, np.newaxis]
        #     sampled_adata.X += noise

        sampled_adatas.append(sampled_adata)
    semisyn_spatial_adata = sc.concat(sampled_adatas, axis=0, )

    #var doesnt copy https://discourse.scverse.org/t/help-with-concat/676/2
    merged_var = pd.concat([x.var for x in sampled_adatas])
    merged_var = merged_var[~merged_var.index.duplicated()]
    semisyn_spatial_adata.var = merged_var.loc[semisyn_spatial_adata.var_names]

    return semisyn_spatial_adata


def _leiden_subclustering_binary_search(adata, cluster_id, target_subclusters):
    cluster_data = adata[adata.obs["leiden"] == cluster_id]
    sc.pp.neighbors(cluster_data, use_rep=f"X_rep_subclustering_{cluster_id}")

    min_resolution = 0.01
    max_resolution = 5.0

    while min_resolution <= max_resolution:
        resolution = (min_resolution + max_resolution) / 2
        sc.tl.leiden(
            cluster_data, resolution=resolution, key_added=f"leiden_sub_{cluster_id}"
        )
        subcluster_labels = cluster_data.obs[f"leiden_sub_{cluster_id}"].unique()

        if len(subcluster_labels) < target_subclusters:
            min_resolution = resolution + 0.01
        elif len(subcluster_labels) > target_subclusters:
            max_resolution = resolution - 0.01
        else:
            break

    print(
        f"Cluster {cluster_id}: arrived at {len(subcluster_labels)} subclusters at resolution {resolution}."
    )
    subcluster_labels = cluster_data.obs[f"leiden_sub_{cluster_id}"].astype(str)
    subcluster_labels = [f"{cluster_id}_sub{label}" for label in subcluster_labels]
    adata.obs.loc[cluster_data.obs.index, "subtype"] = subcluster_labels
    return adata


@click.command()
@click.argument("interaction_csv", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
def main(interaction_csv, output_path):
    random.seed(42)
    np.random.seed(42)

    generate_synthetic_dataset(interaction_csv, output_path)
    print(f"Successfully wrote output to {output_path}")


if __name__ == "__main__":
    main()
