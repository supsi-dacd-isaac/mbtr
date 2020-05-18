#######################################################################################################################
#                                       Quantile loss modeling example
# This example takes data from dataset presented in [1]. It is a much simpler version of what is done in [2], the
# example has only didactic purposes. In particular:
# 1) here we only use endogenous data to model the quantiles, while in [2] we used also NWP signals
#    (also available in [1]) and categorical data, like day of the week and vacations.
# 2) here we only predict quantiles for the first step-ahead, while in [2] we predicted quantiles for 24-hours ahead
#
# [1] "L.Nespoli, V.Medici, K. Lopatichki, F. Sossan, Hierarchical Demand Forecasting Benchmark for the
# Distribution Grid, PSCC 2020, accepted"
# [2] "L.Nespoli, V.Medici, Multivariate Boosted Trees and Applications to Forecasting and Control, 2020, under review"
#######################################################################################################################


import numpy as np
import mbtr.utils as ut
from mbtr.mbtr import MBT
from scipy.linalg import hankel
import matplotlib.pyplot as plt

# --------------------------- Download and format data ----------------------------------------------------------------
# download power data from "Hierarchical Demand Forecasting Benchmark for the Distribution Grid" dataset
# from https://zenodo.org/record/3463137#.XruXbx9fiV7 if needed
power_data = ut.load_dataset()

total_power = power_data['P_mean']['all'].values.reshape(-1, 1)

# down-sample it to 1 hour, bin averages
total_power = np.mean(total_power[:len(total_power)-len(total_power) % 6].reshape(-1, 6), axis=1, keepdims=True)

# embed the signal in a 2-days matrix
total_power = hankel(total_power, np.zeros((25, 1)))[:-25, :]

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
    plt.scatter(25, y_tr[i, :], label='multivariate targets', marker='.')
    plt.xlabel('step ahead [h]')
    plt.ylabel('P [kW]')
    plt.legend(loc='upper right')
    plt.pause(1e-6)
plt.close('all')

# --------------------------- Set up an MBT for quantiles prediction and train it -------------------------------------
alphas = np.linspace(0.05, 0.95, 7)
m = MBT(loss_type='quantile', alphas=alphas, n_boosts=40,
        min_leaf=300, lambda_weights=1e-3).fit(x_tr, y_tr, do_plot=True)

# --------------------------- Predict and plot ------------------------------------------------------------------------

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

