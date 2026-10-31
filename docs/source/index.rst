MBTR documentation
==================

MBTR fits boosted trees whose leaves may return coupled, multivariate, or
structured regression responses, as well as shared-tree Bernoulli logits for
binary and multilabel classification.

.. toctree::
   :maxdepth: 2
   :caption: User documentation

   introduction
   user_guide
   examples
   performance
   studies
   migration
   extending
   mbtr

The public estimators are :class:`mbtr.MBTRegressor` and
:class:`mbtr.MBTClassifier`.  The historical :class:`mbtr.MBT` class remains
as a transition wrapper for v0.1 code.

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
