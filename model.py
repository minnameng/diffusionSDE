# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 20:34:13 2023

@author: mervyns
"""

import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt

np.random.seed(seed=42)

N = 20000
paths = 5000
T = 5
T_vec = np.linspace(0, T, N)
dt = T/N

kappa = 2
theta = 1
sigma = 0.5
std_asy = np.sqrt(sigma**2 / (2*kappa)) # asymptotic standard deviation

X0 = 2 # mit normal verteilung
X = np.zeros((paths, N))
X[:, 0] = X0
W = ss.norm.rvs(loc=0, scale=1, size=(paths, N - 1))

# # Ornstein Uhlenbeck results
# coeff = np.exp(-kappa*dt)
# std_dt = np.sqrt(sigma**2/(2*kappa) * (1 - coeff**2))

# for i in range(N-1):
#     X[:,i+1] = theta + (X[:, i] - theta)*coeff + std_dt * W[:, i]

# Euler Maruyama

for i in range(N-1):
    X[:, i+1] = X[:, i] - kappa*X[:, i]*dt + sigma * np.sqrt(dt) * W[:, i]

N_processes = 10  # number of processes


fig = plt.figure(figsize=(16, 5))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax1.plot(T_vec, X[:N_processes, :].T, linewidth=0.5)
# ax1.plot(T_vec, (theta + std_asy) * np.ones_like(T_vec), label="1 asymptotic std dev", color="black")
# ax1.plot(T_vec, (theta - std_asy) * np.ones_like(T_vec), color="black")
# ax1.plot(T_vec, theta * np.ones_like(T_vec), label="Long term mean")
ax1.legend(loc="upper right")
ax1.set_title(f"{N_processes} OU processes")
ax1.set_xlabel("T")
#ax2.plot(x, pdf_fitted, color="r", label="Normal density")
ax2.hist(X[:, -1], density=True, bins=50, facecolor="LightBlue", label="Frequency of X(T)")
ax2.legend()
ax2.set_title("Histogram vs Normal distribution")
ax2.set_xlabel("X(T)")
plt.show()


Y0 = ss.norm.rvs(loc=0, scale=ss.norm.fit(X[:,-1])[1], size=paths)
WY = ss.norm.rvs(loc=0, scale=1, size=(paths, N - 1))
Y = np.zeros((paths, N))
Y[:,0] = Y0
std_vec = np.zeros(N)

for i in range(N-100):
    # mean, std = ss.norm.fit(X[:, i+1])
    mean = np.exp(-kappa*T_vec[N-1-i]) * X0
    std = sigma**2/(2*kappa) *(1-np.exp(-2*kappa*T_vec[N-1-i]))
    Y[:, i+1] = Y[:,i] - (kappa*Y[:,i]+sigma**2/std *(Y[:,i]-mean))*dt + sigma*np.sqrt(dt)*WY[:,i]
    std_vec[i] = 1/std

fig = plt.figure(figsize=(16, 5))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax1.plot(T_vec, Y[:N_processes, :].T, linewidth=0.5)
ax2.hist(Y[:, N-100], density=True, bins=50, facecolor="LightBlue")
plt.show()

# plt.plot(T_vec[:15000], 1./std_vec[:10000])
# plt.show()