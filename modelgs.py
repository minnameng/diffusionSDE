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
sigma = 1
std_asy = np.sqrt(sigma**2 / (2*kappa)) # asymptotic standard deviation

# initial distribution
mu = 2
sig2 = 2

X0 = ss.norm.rvs(loc=mu, scale=np.sqrt(sig2), size=paths)
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
x = np.linspace(-3, 3, 300)
ax2.plot(x, ss.norm.pdf(x,loc=0,scale=std_asy), color="r", label="Normal density")
ax2.hist(X[:, -1], density=True, bins=50, facecolor="LightBlue", label="Frequency of X(T)")
ax2.legend()
ax2.set_title("Histogram vs Normal distribution")
ax2.set_xlabel("X(T)")
plt.show()

print(std_asy)
print(ss.norm.fit(X[:,-1]))

Y0 = ss.norm.rvs(loc=0, scale=std_asy, size=paths)
# Y0 = ss.norm.rvs(loc=0, scale=ss.norm.fit(X[:,-1])[1], size=paths)
WY = ss.norm.rvs(loc=0, scale=1, size=(paths, N - 1))
Y = np.zeros((paths, N))
Y[:,0] = Y0
Y_error = np.zeros((paths, N))
Y_error[:,0] = Y0
std_vec = np.zeros(N)
t_lower = 10


mt = np.exp(-kappa*T_vec[::-1]) # reverse time line
sigt2 = sigma**2/(2*kappa) * (1-mt**2)

mean = mt * mu
std2 = sigt2 + sig2*(mt**2)

print(mean[:5])
print(std2[:5])
for i in range(N-1):
    # mean, std = ss.norm.fit(X[:, i+1])
    Y[:, i+1] = Y[:,i] + (kappa*Y[:,i]-sigma**2/std2[i] *(Y[:,i]-mean[i]))*dt + sigma*np.sqrt(dt)*WY[:,i]
    Y_error[:, i+1] = Y_error[:,i] + (kappa*Y_error[:,i]-sigma**2 * (1/std2[i] *(Y_error[:,i]-mean[i]) +3))*dt + sigma*np.sqrt(dt)*WY[:,i]
    # std_vec[i] = 1/std2

fig = plt.figure(figsize=(16, 5))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
#ax3 = fig.add_subplot(133)
#ax1.plot(T_vec, Y[:N_processes, :].T, linewidth=0.5)
x0 = np.linspace(-2,6,300)
ax1.plot(x0, ss.norm.pdf(x0,loc=mu,scale=np.sqrt(sig2)), color="r", label="Normal density")
ax1.hist(Y[:,-1], density=True, bins=50, facecolor="LightBlue")
ax2.plot(x0, ss.norm.pdf(x0,loc=mu,scale=np.sqrt(sig2)), color="r", label="Normal density")
ax2.hist(Y_error[:,-1], density=True, bins=50, facecolor="LightBlue")
plt.show()
