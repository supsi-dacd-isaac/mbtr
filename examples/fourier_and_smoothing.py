#######################################################################################################################
#                                   Multivariate smoothing and Fourier modeling example
# This example takes data from dataset presented in [1]. It is a much simpler version of what is done in [2], the
# example has only didactic purposes. In particular in [2]:
# 1) we used exogenous covariates, as NWP variables, day of week, vacations
# 2) we trained a MBT for all the series in the dataset
# 3) we used a different embedding time (including observations up to one week before)
# in [2] we used also NWP signals (also available in [1]).
# [1] "L.Nespoli, V.Medici, K. Lopatichki, F. Sossan, Hierarchical Demand Forecasting Benchmark for the
# Distribution Grid, PSCC 2020, accepted"
# [2] "L.Nespoli, V.Medici, Multivariate Boosted Trees and Applications to Forecasting and Control, 2020, under review"
#######################################################################################################################


import numpy as np
import mbtr.utils as ut
from scipy.linalg import hankel
import matplotlib.pyplot as plt
from mbtr.mbtr import MBT

# --------------------------- Download and format data ----------------------------------------------------------------
# download power data from "Hierarchical Demand Forecasting Benchmark for the Distribution Grid" dataset
# from https://zenodo.org/record/3463137#.XruXbx9fiV7 if needed
power_data = ut.load_dataset()

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


# --------------------------- Set up an MBT with smooth regularization and fit it -------------------------------------
print('#'*20 + '    Fitting MBT with smooth loss   ' + '#'*20)
m_sm = MBT(loss_type='time_smoother', lambda_smooth=1, n_boosts=30,
           min_leaf=300, lambda_weights=1e-3).fit(x_tr, y_tr, do_plot=True)
y_hat_sm = m_sm.predict(x_te)


# --------------------------- Set up 2 MBT with Fourier response  ------------------------------------------------------
print('#'*20 + '    Fitting MBT with Fourier loss and 3 harmonics    ' + '#'*20)

m_fou_3 = MBT(loss_type='fourier', n_harmonics=3, n_boosts=30,
              min_leaf=300, lambda_weights=1e-3).fit(x_tr, y_tr, do_plot=True)
y_hat_fou_3 = m_fou_3.predict(x_te)

print('#'*20 + '    Fitting MBT with Fourier loss and 5 harmonics    ' + '#'*20)

m_fou_5 = MBT(loss_type='fourier', n_harmonics=5, n_boosts=30,  min_leaf=300,
              lambda_weights=1e-3).fit(x_tr, y_tr, do_plot=True)
y_hat_fou_5 = m_fou_5.predict(x_te)


# --------------------------- plot first 150 horizons -----------------------------------------------------------------
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

# --------------------------- Print mean-horizon RMSEs form the models ------------------------------------------------

mean_rmse = lambda x, y: np.mean(np.mean((x - y) ** 2, axis=1) ** 0.5)
rmse_sm = mean_rmse(y_te, y_hat_sm)
rmse_fou_3 = mean_rmse(y_te, y_hat_fou_3)
rmse_fou_5 = mean_rmse(y_te, y_hat_fou_5)


print('#' * 20 + '    Mean-horizon RMSEs  ' + '#' * 20)
[print('{}: {:0.2e}'.format(n, s)) for n, s in zip(['smoother', 'fourier-3', 'fourier-5'], [rmse_sm, rmse_fou_3, rmse_fou_5])]