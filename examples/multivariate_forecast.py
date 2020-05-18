#######################################################################################################################
#                                       Multivariate modeling example
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
total_power = np.mean(total_power[:len(total_power)-len(total_power)%6].reshape(-1, 6), axis=1, keepdims=True)

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

# --------------------------- Set up an MBT and fit it --------------------------------------------------------------
print('#'*20 + '    Fitting MBT with mse loss   ' + '#'*20)
m = MBT(n_boosts=30,  min_leaf=100, lambda_weights=1e-3).fit(x_tr, y_tr, do_plot=True)
y_hat = m.predict(x_te)


# --------------------------- Set up 24 MISO LightGBM and fit it ------------------------------------------------------
print('#'*20 + '    Fitting 24 MISO LightGBMs   ' + '#'*20)

m_lgb = ut.LightGBMMISO(30).fit(x_tr, y_tr)
y_hat_lgb = m_lgb.predict(x_te)


# --------------------------- Set up a linear-MBT and fit it ----------------------------------------------------------
# The MBT chooses splits based on the previous day mean, min, max, first and last values. It then fits a linear
# model inside the leaves
print('#'*20 + '    Fitting a linear-response MBT    ' + '#'*20)

x_build = np.hstack([np.mean(x, axis=1,keepdims=True), np.max(x,axis=1, keepdims=True),
                     np.min(x, axis=1, keepdims=True), x[:,[0,23]]])
x_build_tr, x_build_te = [x_build[:n_tr, :],x_build[n_tr:, :]]
m_lin = MBT(loss_type='linear_regression', n_boosts=30,  min_leaf=1500,
            lambda_weights=1e-3).fit(x_build_tr, y_tr,x_lr=x_tr, do_plot=True)
y_hat_lin = m_lin.predict(x_build_te, x_lr=x_te)

# --------------------------- plot first 150 horizons -----------------------------------------------------------------
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

# --------------------------- Print mean-horizon RMSEs form the models ------------------------------------------------

mean_rmse = lambda x,y : np.mean(np.mean((x-y)**2, axis=1)**0.5)
rmse_mbt = mean_rmse(y_te, y_hat)
rmse_lgb = mean_rmse(y_te, y_hat_lgb)
rmse_mbt_lin = mean_rmse(y_te, y_hat_lin)

print('#'*20 + '    Mean-horizon RMSEs  ' + '#'*20 )
[print('{}: {:0.2e}'.format(n, s)) for n, s in zip(['mbtr', 'lgb', 'mbtr-lin'], [rmse_mbt, rmse_lgb, rmse_mbt_lin])]