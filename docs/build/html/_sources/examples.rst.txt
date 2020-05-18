Examples
=============
Some didactic examples can be found in the :code:`mbtr/examples` dir. All the examples are based on the
`Hierarchical Demand Forecasting Benchmark`_, which is downloaded at the beginning of the examples. The dataset is
downloaded only once, successive calls to :meth:`mbtr.ut.download_dataset` will only read the locally downloaded file.

.. _Hierarchical Demand Forecasting Benchmark: https://zenodo.org/record/3463137#.XsIwGR9fiV7

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


Multivariate forecasts and linear regression
********************************************
The :code:`examples/multivariate_forecast.py` shows an example of usage of the :class:`mbtr.losses.MSE` and
:class:`mbtr.losses.LinRegLoss` losses.

We start creating the training and test set. We use the P_mean signal as a target, and we want to predict the next day
ahead only using past values from the same time series. We start downsampling the signal to one hour, and then we embed
it in a 2-days matrix. The first 24 columns refers to the previous day in time, and are the signals which will be used
to predict the last 24 columns:

.. code-block:: python

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


we can perform a visual check on the features and target signals:

.. code-block:: python

    # visual check on the first 50 samples of features and targets
    plt.figure()
    for i in range(50):
        plt.cla()
        plt.plot(np.arange(24), x_tr[i,:], label='features')
        plt.plot(np.arange(24) + 24, y_tr[i, :], label='multivariate targets')
        plt.xlabel('step ahead [h]')
        plt.ylabel('P [kW]')
        plt.legend(loc='upper right')
        plt.pause(1e-6)
    plt.close('all')

Now we are ready to fit an MBT instance. We start fitting a mean squared error loss function:

.. code-block:: python

    print('#'*20 + '    Fitting MBT with mse loss   ' + '#'*20)
    m = MBT(n_boosts=30,  min_leaf=100, lambda_weights=1e-3).fit(x_tr, y_tr, do_plot=True)
    y_hat = m.predict(x_te)

As a comparison, we also fit also 24 different :code:`LightGBM` instances with the utility class
:class:`mbtr.ut.LightGBMMISO` :

.. code-block:: python

    print('#'*20 + '    Fitting 24 MISO LightGBMs   ' + '#'*20)

    m_lgb = ut.LightGBMMISO(30).fit(x_tr, y_tr)
    y_hat_lgb = m_lgb.predict(x_te)

As a last comparison, we fit a linear response MBT, using the :class:`mbtr.losses.LinRegLoss`. This class requires as
additional input a set of features for fitting the linear response inside each leaf. In order to reduce the
computational time, we only use the mean, maximum, minimum and the first and last values of the original regressors
matrix :code:`x` as features for finding the best splits of the trees.

.. code-block:: python

    x_build = np.hstack([np.mean(x, axis=1,keepdims=True), np.max(x,axis=1, keepdims=True),
                         np.min(x, axis=1, keepdims=True), x[:,[0,23]]])
    x_build_tr, x_build_te = [x_build[:n_tr, :],x_build[n_tr:, :]]
    m_lin = MBT(loss_type='linear_regression', n_boosts=30,  min_leaf=1500,
                lambda_weights=1e-3).fit(x_build_tr, y_tr,x_lr=x_tr, do_plot=True)
    y_hat_lin = m_lin.predict(x_build_te, x_lr=x_te)


We can now plot some results:

.. code-block:: python

    for i in range(150):
        plt.cla()
        plt.plot(np.arange(24), y_te[i,:], label='test')
        plt.plot(y_hat[i, :], '--', label='mbtr')
        plt.plot(y_hat_lgb[i, :], '--', label='lgb')
        plt.plot(y_hat_lin[i, :], '--', label='mbtr-lin')
        plt.xlabel('step ahead [h]')
        plt.ylabel('P [kW]')
        plt.legend(loc='upper right')
        plt.pause(1e-6)

and compare the models in term of RMSE:

.. code-block:: python

    mean_rmse = lambda x,y : np.mean(np.mean((x-y)**2, axis=1)**0.5)
    rmse_mbt = mean_rmse(y_te, y_hat)
    rmse_lgb = mean_rmse(y_te, y_hat_lgb)
    rmse_mbt_lin = mean_rmse(y_te, y_hat_lin)

    print('#'*20 + '    Mean-horizon RMSEs  ' + '#'*20 )
    [print('{}: {:0.2e}'.format(n, s)) for n, s in zip(['mbtr', 'lgb', 'mbtr-lin'], [rmse_mbt, rmse_lgb, rmse_mbt_lin])]

Time smoothing and Fourier regression
*************************************
The :code:`examples/fourier_and_smoothing.py` shows an example of usage of the :class:`mbtr.losses.TimeSmoother` and
:class:`mbtr.losses.FourierLoss` losses.
The first part of the code is identical to the one used in the :code:`examples/multivariate_forecast.py` example; we
download the dataset and create the training and test sets. We now use the :code:`time_smoother` loss function,
which penalize the second order discrete derivative of the response function:

.. code-block:: python

    print('#'*20 + '    Fitting MBT with smooth loss   ' + '#'*20)
    m_sm = MBT(loss_type='time_smoother', lambda_smooth=1, n_boosts=30,
               min_leaf=300, lambda_weights=1e-3).fit(x_tr, y_tr, do_plot=True)
    y_hat_sm = m_sm.predict(x_te)


Keeping all the other MBT parameters unchanged, we can fit two Fourier losses with different number of harmonics:

.. code-block:: python

    print('#'*20 + '    Fitting MBT with Fourier loss and 3 harmonics    ' + '#'*20)

    m_fou_3 = MBT(loss_type='fourier', n_harmonics=3, n_boosts=30,
                  min_leaf=300, lambda_weights=1e-3).fit(x_tr, y_tr, do_plot=True)
    y_hat_fou_3 = m_fou_3.predict(x_te)

    print('#'*20 + '    Fitting MBT with Fourier loss and 5 harmonics    ' + '#'*20)

    m_fou_5 = MBT(loss_type='fourier', n_harmonics=5, n_boosts=30,  min_leaf=300,
                  lambda_weights=1e-3).fit(x_tr, y_tr, do_plot=True)
    y_hat_fou_5 = m_fou_5.predict(x_te)


We can now plot some results from the different fitted losses:

.. code-block:: python

    for i in range(150):
        plt.cla()
        plt.plot(np.arange(24), y_te[i,:], label='test')
        plt.plot(y_hat_sm[i, :], '--', label='time-smoother')
        plt.plot(y_hat_fou_3[i, :], '--', label='fourier-3')
        plt.plot(y_hat_fou_5[i, :], '--', label='fourier-5')

        plt.xlabel('step ahead [h]')
        plt.ylabel('P [kW]')
        plt.legend(loc='upper right')
        plt.pause(1e-6)

Finally, we can compare the models in term of RMSE:

.. code-block:: python

    mean_rmse = lambda x, y: np.mean(np.mean((x - y) ** 2, axis=1) ** 0.5)
    rmse_sm = mean_rmse(y_te, y_hat_sm)
    rmse_fou_3 = mean_rmse(y_te, y_hat_fou_3)
    rmse_fou_5 = mean_rmse(y_te, y_hat_fou_5)


    print('#' * 20 + '    Mean-horizon RMSEs  ' + '#' * 20)
    [print('{}: {:0.2e}'.format(n, s)) for n, s in zip(['smoother', 'fourier-3', 'fourier-5'], [rmse_sm, rmse_fou_3, rmse_fou_5])]



Quantile loss
*************
The :code:`examples/quantiles.py` shows an example of usage of the :class:`mbtr.losses.QuantileLoss` loss.
In this example we aim at predicting the quantiles of the next step ahead, using the previous 24 hours of the signal as
covariates. After downloading the dataset as described in the previous example, we build the training and test sets:

.. code-block:: python

    # embed the signal in a 2-days matrix
    total_power = hankel(total_power, np.zeros((25, 1)))[:-25, :]

    # create feature matrix and target for the training and test sets
    x = total_power[:, :24]
    y = total_power[:, 24:]
    n_tr = int(len(x)*0.8)
    x_tr, y_tr, x_te, y_te = [x[:n_tr, :], y[:n_tr, :], x[n_tr:, :], y[n_tr:, :]]

we plot some training instances of the features and the target to have a visual check:

.. code-block:: python

    plt.figure()
    for i in range(50):
        plt.cla()
        plt.plot(np.arange(24), x_tr[i,:], label='features')
        plt.scatter(25, y_tr[i, :], label='multivariate targets', marker='.')
        plt.xlabel('step ahead [h]')
        plt.ylabel('P [kW]')
        plt.legend(loc='upper right')
        plt.pause(1e-6)
    plt.close('all')

Finally, we can train a :code:`MBT` instance with a :class:`mbtr.losses.QuantileLoss` loss. Note that this loss requires
the :code:`alphas` additional parameter. This is an array of quantiles to be fitted:

.. code-block:: python

    alphas = np.linspace(0.05, 0.95, 7)
    m = MBT(loss_type='quantile', alphas=alphas, n_boosts=40,
            min_leaf=300, lambda_weights=1e-3).fit(x_tr, y_tr, do_plot=True)

At last, we can plot some predictions for the required quantiles:

.. code-block:: python

    y_hat = m.predict(x_te)
    fig,ax = plt.subplots(1)
    n_q = y_hat.shape[1]
    n_sa = y_te.shape[1]
    n_plot = 300
    colors = plt.get_cmap('plasma', int(n_q))
    for fl in np.arange(np.floor(n_q / 2), dtype=int):
        q_low = np.squeeze(y_hat[:n_plot, fl])
        q_up = np.squeeze(y_hat[:n_plot, n_q - fl - 1])
        x = np.arange(len(q_low))
        ax.fill_between(x, q_low, q_up, color=colors(fl), alpha=0.1 + 0.6*fl/n_q, linewidth=0.0)
    plt.plot(y_te[:n_plot], linewidth=2)
    plt.xlabel('step [h]')
    plt.ylabel('P [kW]')
    plt.title('Quantiles on first 300 samples')




