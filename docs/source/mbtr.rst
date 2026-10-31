API reference
=============

Estimator
---------

.. autoclass:: mbtr.MBTRegressor
   :members:
   :inherited-members:
   :show-inheritance:

.. autoclass:: mbtr.MBTClassifier
   :members:
   :show-inheritance:

Compatibility estimator
-----------------------

.. autoclass:: mbtr.MBT
   :members:
   :show-inheritance:

Objectives
----------

.. autoclass:: mbtr.losses.Loss
   :members:

.. autoclass:: mbtr.losses.MSE

.. autoclass:: mbtr.losses.BinaryLogLoss

.. autoclass:: mbtr.losses.TimeSmoother

.. autoclass:: mbtr.losses.FourierLoss

.. autoclass:: mbtr.losses.LatentVariable

.. autoclass:: mbtr.losses.LinRegLoss

.. autoclass:: mbtr.losses.QuantileLoss

.. autoclass:: mbtr.losses.QuadraticQuantileLoss

.. autoclass:: mbtr.losses.PseudoHuberQuantileLoss

.. autoclass:: mbtr.losses.LeafSolution
   :members:

Optional utilities
------------------

.. autofunction:: mbtr.utils.load_dataset

.. autofunction:: mbtr.utils.download_dataset

.. autoclass:: mbtr.utils.LightGBMMISO
   :members:

.. autofunction:: mbtr.utils.set_figure
