Overview
========

LYNX integrates two paired spatial modalities measured over the same tissue
and infers how molecular state changes across space.

Pipeline
--------

A typical LYNX analysis follows the same five stages regardless of the
modality pairing:

1. **Load and pair the modalities.** A *reference* modality (usually spatial
   transcriptomics) and a *query* modality (metabolomics, histology image
   patches, or protein abundance) are loaded into ``AnnData`` objects that
   share a spatial coordinate system.

2. **Build a heterogeneous graph.** :class:`lynx.dataset.HeteroDataset` ties
   the two modalities into a single spatial graph, partitioned into subgraphs
   for scalable training.

3. **Train the model.** :class:`lynx.model.HeteroAttnVGAE`, a heterogeneous
   attention variational graph auto-encoder, learns a shared latent embedding
   (``X_z``) and reconstructs the reference features. Cell–cell interaction
   inference can be switched on through the model config.

4. **Infer a trajectory.** From the latent embedding, LYNX fits either a
   principal **curve** (:func:`lynx.trajectory.get_curve`) for monotonic
   gradients or a principal **tree** (:func:`lynx.trajectory.get_tree`) for
   a tree-structure inference given more complicated manifold, then assigns a continuous pseudotime (sorted spatial coordinates) ``t`` with
   :func:`lynx.trajectory.compute_pseudotime`.

5. **Analyse zones and interactions.** The continuous gradient is discretised
   into tissue zones (:func:`lynx.utils.get_zonations`,
   :func:`lynx.utils.get_zonation_features`), and inferred cell–cell
   interactions are summarised and visualised
   (:func:`lynx.plot.summarize_cell_interaction`,
   :func:`lynx.plot.netVisual_circle`).

Import namespace
----------------

The public API is exposed under a single ``lynx`` namespace::

    import lynx

    graph_data = lynx.dataset.HeteroDataset(...)
    model = lynx.model.HeteroAttnVGAE(...)
    lynx.trajectory.get_curve(adata)
    lynx.plot.disp_trajectory(adata)

See the :doc:`api/index` for the full surface.
