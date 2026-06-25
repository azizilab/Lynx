Tutorials
=========

Each tutorial runs the full LYNX pipeline on a different multi-omic spatial
dataset. The training step is shown as code; the downstream analysis and plots
load a pre-saved LYNX result snapshot, so the figures render without re-running
the (GPU-bound) model.

- :doc:`liver` — Xenium spatial transcriptomics + DESI metabolomics; principal
  **curve** trajectory (periportal → pericentral zonation) with cell–cell
  interactions.
- :doc:`breast` — Xenium spatial transcriptomics + H&E histology; principal
  **tree** trajectory (DCIS → invasive branching).
- :doc:`thymus` — spatial RNA + protein (CITE-seq); principal **curve**
  trajectory (cortex → medulla axis), the most minimal example.

.. toctree::
   :maxdepth: 1

   liver
   breast
   thymus
