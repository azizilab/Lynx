LYNX
====

**LYNX** is a method for spatially-resolved gradient inference with multi-omic
integration. It learns a shared latent representation from paired spatial
modalities (e.g. spatial transcriptomics & metabolomics,
histology, or protein abundance) using variational
graph auto-encoder on a spatial hetero-graph, then infers continuous spatial gradients, discrete tissue
zones, and cell–cell interactions on top of that representation.

Get started
-----------

- :doc:`overview` — what LYNX does and how the pipeline fits together.
- :doc:`installation` — set up LYNX with conda (preferred) or pip.
- :doc:`tutorials/index` — worked examples across three multi-omic datasets.
- :doc:`api/index` — the ``lynx`` public API.

Tutorials at a glance
---------------------

.. list-table::
   :header-rows: 1
   :widths: 25 50

   * - Tutorial
     - Modalities
   * - :doc:`tutorials/liver`
     - Xenium (transcriptomics) + DESI (metabolomics)
   * - :doc:`tutorials/breast`
     - Xenium (transcriptomics) + histology
   * - :doc:`tutorials/thymus`
     - Spatial RNA + protein (Stereo-CITE-seq)


.. toctree::
   :maxdepth: 1
   :hidden:

   overview
   installation
   tutorials/index
   api/index
