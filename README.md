# LYNX

[![Documentation Status](https://readthedocs.org/projects/lynx/badge/?version=latest)](https://lynx.readthedocs.io/en/latest/)

Spatial trajectory inference with multi-omic integration.

LYNX learns a shared latent representation from paired spatial modalities
(e.g. spatial transcriptomics together with metabolomics, histology, or protein
abundance) using a heterogeneous attention variational graph auto-encoder, then
infers continuous spatial gradients, discrete tissue zones, and cell–cell
interactions on top of that representation.

📖 **Documentation:** https://lynx.readthedocs.io

## Installation

LYNX targets **Python 3.9.5**. Conda is recommended; a pip-only track is
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

## Tutorials

One worked example per multi-omic pairing (see the
[tutorials](https://lynx.readthedocs.io/en/latest/tutorials/index.html)):

| Tutorial | Modalities | Trajectory |
| --- | --- | --- |
| [Liver](https://lynx.readthedocs.io/en/latest/tutorials/liver.html) | Xenium transcriptomics + DESI metabolomics | principal curve |
| [Breast](https://lynx.readthedocs.io/en/latest/tutorials/breast.html) | Xenium transcriptomics + H&E histology | principal tree |
| [Thymus](https://lynx.readthedocs.io/en/latest/tutorials/thymus.html) | spatial RNA + protein (CITE-seq) | principal curve |

## Quickstart

```python
import lynx

graph_data = lynx.dataset.HeteroDataset(adatas_ref=adata_ref, adatas_query=adata_query)
model = lynx.model.HeteroAttnVGAE(model_configs)
model.fit(graph_data, train_configs)

lynx.trajectory.get_curve(adata_ref)
lynx.plot.disp_trajectory(adata_ref)
```

See the [API reference](https://lynx.readthedocs.io/en/latest/api/index.html) for
the full surface.
