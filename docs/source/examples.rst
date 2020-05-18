Examples
=============
Some didactic examples can be found in the `mbtr/examples` dir. All the examples are based on the
`Hierarchical Demand Forecasting Benchmark`_, which is downloaded at the beginning of the examples. The dataset is
downloaded only once, successive calls to :meth:`mbr.ut.download_dataset`
.. code-block:: python

    import numpy as np
    import mbtr.utils as ut
    from scipy.linalg import hankel
    import matplotlib.pyplot as plt
    from mbtr.mbtr import MBT

    # --------------------------- Download and format data ----------------------------------------------------------------
    # download power data from "Hierarchical Demand Forecasting Benchmark for the Distribution Grid" dataset
    # from https://zenodo.org/record/3463137#.XruXbx9fiV7 if needed
    power_data = ut.download_dataset()

.. _Hierarchical Demand Forecasting Benchmark: https://zenodo.org/record/3463137#.XsIwGR9fiV7

Time smoothing and Fourier regression
*************************************


    total_power = power_data['P_mean']['all'].values.reshape(-1,1)
    # down-sample it to 1 hour, bin averages
    total_power = np.mean(total_power[:len(total_power)-len(total_power) % 6].reshape(-1, 6), axis=1, keepdims=True)

    # embed the signal in a 2-days matrix
    total_power = hankel(total_power, np.zeros((2 * 24, 1)))

    # create feature matrix and target for the training and test sets
    x = total_power[:, :24]
    y = total_power[:, 24:]
    n_tr = int(len(x)*0.8)

    x_tr, y_tr, x_te, y_te = [x[:n_tr, :], y[:n_tr, :], x[n_tr:, :], y[n_tr:, :]]
