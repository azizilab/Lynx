Association tests
=================

Statistical tests for cell–cell interactions and trajectory associations.

``test_cci`` returns the interaction strength of the significant sender → receiver pairs and their significance, by the
test of the fitted model's version. For the revised model (the default) this is the C × C level of
``get_test_interactions``, which tests every level with one scheme: C × C (does sender type *a* change receiver type
*b*), C × C × G (which genes of *b*), C × C × P (which user-given programs) and, for niche fields or group labels,
K × C, K × C × G and K × C × P.

.. currentmodule:: lynx

.. autosummary::
   :toctree: generated
   :nosignatures:

   test_assoc.test_cci
   test_assoc.get_test_interactions
   test_assoc.test_lr
   test_assoc.cxcxg_mask
   test_assoc.fit_leak
   test_assoc.surrogate_fields
   test_assoc.consensus
   test_assoc.test_trajectory
   test_assoc.get_test_associations
