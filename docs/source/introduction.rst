Introduction
============

MBTR extends gradient-boosted trees to targets with multiple, related outputs.
A leaf can return a constant regression vector, a smooth response, a Fourier
expansion, a response constrained by a latent linear map, a set of quantiles,
a ridge-linear function of separate response features, or a vector of
Bernoulli-logit updates for multilabel classification.

The v0.2 implementation keeps signed gradient and curvature statistics and
uses the same local solver for both the candidate response and its split gain.
In the common quadratic form,

.. math::

   Q_l(\theta) = -b_l^T\theta + \frac{1}{2}\theta^T A_l\theta + \gamma,

the leaf parameters and gain are

.. math::

   \theta_l^* = A_l^{-1}b_l,
   \qquad
   \operatorname{gain}_l = \frac{1}{2}b_l^T\theta_l^*.

This contract is important for structured objectives: applying an inverse
coupled Hessian to coordinate-wise squared gradients would discard the
cross-terms that define the multivariate model.

An explicitly documented post-structure refit is a separate operation.  The
published quantile objectives use smooth derivatives to choose partitions,
then may replace terminal responses with exact empirical pinball minimizers.
Their public metric remains pinball loss; this intentional hybrid does not
change the quadratic-gain contract used while searching for splits.

Installation
------------

Install the core package from PyPI:

.. code-block:: bash

   python -m pip install mbtr

Optional groups keep the core installation small:

.. code-block:: bash

   python -m pip install "mbtr[fast]"
   python -m pip install "mbtr[plot]"
   python -m pip install "mbtr[examples]"

The ``examples`` group is needed only for the external benchmark dataset and
the optional LightGBM comparison.

Scope and guarantees
--------------------

* NumPy is the authoritative reference engine.
* An accelerated engine must pass numerical parity tests before it can become
  a default.
* ``min_samples_leaf`` applies to both children of every accepted split.
* ``random_state`` makes convenience validation splits reproducible.
* Quantile predictions explicitly retain output and quantile dimensions.
* Classification targets are explicit binary indicators; one-hot multiclass
  targets are not silently treated as independent labels.
* Missing values are rejected unless a documented missing-value strategy is
  selected; they are never routed silently.

Citation
--------

If MBTR supports your research, cite the published JMLR paper:

.. code-block:: bibtex

   @article{JMLR:v23:21-0247,
     author  = {Lorenzo Nespoli and Vasco Medici},
     title   = {Multivariate Boosted Trees and Applications to Forecasting and Control},
     journal = {Journal of Machine Learning Research},
     year    = {2022},
     volume  = {23},
     number  = {246},
     pages   = {1--47},
     url     = {https://jmlr.org/papers/v23/21-0247.html}
   }

See the `paper landing page <https://www.jmlr.org/papers/v23/21-0247.html>`_
for the article and BibTeX source.

Acknowledgments
---------------

The authors thank the Swiss Federal Office of Energy and the Swiss Competence
Center for Energy Research — Future Swiss Electrical Infrastructure for their
financial and technical support.
