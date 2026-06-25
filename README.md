# LYNX: Spatial gradient inference with multi-modal integration.

[![Documentation Status](https://readthedocs.org/projects/lynx/badge/?version=latest)](https://lynx.readthedocs.io/en/latest/)

LYNX is a deep generative model that learns a shared latent representation from paired spatial modalities (e.g., transcriptomics, proteomics, metabolomics, or histology) in adjacent tissue sections. It characterizes microenvironmental gradients reflected by cell-state transitions, signaling pathways, and physiological functions in healthy and diseased tissues (e.g., morphogenesis, tumorigenesis). From this latent space, users can perform downstream tasks including phenotype & feature dynamics and localized changes of cell-cell interactions along the inferred spatial gradients.

📖 **Full Documentation:** https://lynx.readthedocs.io

## Installation

LYNX is implemented and testedwwith **Python 3.9.5**. Conda is recommended; a pip-only track is
available as a fallback. The PyTorch Geometric companion wheels must match your
`torch` + CUDA build.

```bash
conda env create -f environment.yml
conda activate lynx

# PyG companion wheels (match torch 2.3.1 + your CUDA):
pip install torch-sparse==0.6.18 torch-geometric==2.6.1 \
    -f https://data.pyg.org/whl/torch-2.3.1+cu121.html

pip install -e .
```

See the [installation guide](https://lynx.readthedocs.io/en/latest/installation.html)
for the pip fallback and details.

## Quickstart

```python
import lynx
import scanpy as sc

# Load paired spatial data
adata_ref = sc.read_h5ad('#path to primary modality')  # primary modality (`reference`)
adata_query = sc.read_h5ad('#path to auxiliary modality') # auxiliary / secondary modality (`query`)
graph_data = lynx.dataset.HeteroDatase(
    adatas_ref=adata_ref, 
    adatas_query=adata_query
)

# Configure & train LYNX model
model_configs = lynx.config.set_model_configs(graph_data)
model = lynx.model.HeteroAttnVGAE(model_configs)
model.fit(graph_data, train_configs)

# Infer spatial gradient
lynx.trajectory.get_curve(adata_primary)
lynx.plot.disp_trajectory(adata_ref)
```

See the [API reference](https://lynx.readthedocs.io/en/latest/api/index.html) for
the full surface.

## Tutorials

Current working examples with multi-modal applications (see the
[tutorials](https://lynx.readthedocs.io/en/latest/tutorials/index.html)):

| Tutorial | Modalities | Applications |
| --- | --- | --- |
| [Liver](https://lynx.readthedocs.io/en/latest/tutorials/liver.html) | transcriptomics (Xenium) + metabolomics (DESI)| portal-central axis with liver zonation |
| [Breast](https://lynx.readthedocs.io/en/latest/tutorials/breast.html) | transcriptomics (Xenium) + H&E | spatial gradients from immune niche to tumor states |
| [Thymus](https://lynx.readthedocs.io/en/latest/tutorials/thymus.html) | spatial RNA + protein (Stereo-CITE-seq) | Corto-Medullary Axis  |


## Repository structure

```
├── models/       # Core model implementation
├── util/         # util functions for IO, plotting, etc.
├── breast/       # breast application scripts (RNA + H&E)
├── liver/        # liver application scripts (RNA + metabolomics)
├── thymus/       # thymus application scripts (RNA + protein)
├── benchmarks/   # Reproducibility for benchmarks + model ablation
│   ├── breast/, liver/, thymus/   # per-application baseline scripts
│   ├── ablation/                  # model ablation studies
│   └── run_benchmarks.sh          # full automatic run
├── docs/         # documentation, API & tutorial
├── data/         # Raw, intermediate or processed data (gitignored)
├── results/      # Analysis results, saved data, etc. (gitignored)
└── tests/        # Unit tests (import / IO / dataset)
```
