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
T = 1
T_vec = np.linspace(0, T, N)
dt = T/N

kappa = 2
# theta = 1
sigma = 1
std_asy = np.sqrt(sigma**2 / (2*kappa)) # asymptotic standard deviation

# initial distribution
mu = [2, -2]
sig2 = [1, 1]
coef = [1/3, 2/3]

def valuet(drift, diffusion, t):
    mt = np.exp(-drift*t[::-1]) # reverse time line
    sigt2 = diffusion**2/(2*drift) * (1-mt**2)
    return mt, sigt2

def distribution(mt, sigt2, mu, sig2):
    mean = np.kron(mu,mt).reshape((len(mu),len(mt)))
    std2 = sigt2 + np.kron(sig2,mt**2).reshape((len(sig2),len(mt)))
    return mean, std2

def score(y, mean, std2):
    return - (y-mean)/std2

def euler(X0, func, sigma, paths, T_vec, t_lower=1):
    N = len(T_vec)
    dt = T_vec[-1]/N
    X = np.zeros((paths, N))
    X[:, 0] = X0
    W = ss.norm.rvs(loc=0, scale=1, size=(paths, N - 1))
    
    for i in range(N-t_lower):
        X[:, i+1] = X[:, i] + func(X[:, i])*dt + sigma * np.sqrt(dt) * W[:, i]
    
    return X

def plotres(T_vec, X, mean, std_asy, lower, upper, N_processes):
    fig = plt.figure(figsize=(16, 5))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.plot(T_vec, X[:N_processes, :].T, linewidth=0.5)
    ax1.plot(T_vec, (mean+std_asy) * np.ones_like(T_vec), label="1 asymptotic std dev", color="black")
    ax1.plot(T_vec, (mean-std_asy) * np.ones_like(T_vec), color="black")
    ax1.plot(T_vec, mean * np.ones_like(T_vec), label="Long term mean")
    ax1.legend(loc="upper right")
    ax1.set_title(f"{N_processes} OU processes")
    ax1.set_xlabel("T")
    x = np.linspace(lower, upper, 300)
    ax2.plot(x, ss.norm.pdf(x,loc=mean,scale=std_asy), color="r", label="Normal density")
    ax2.hist(X[:, -1], density=True, bins=50, facecolor="LightBlue", label="Frequency of X(T)")
    ax2.legend()
    ax2.set_title("Histogram vs Normal distribution")
    ax2.set_xlabel("X(T)")
    plt.show()

def plothist(sample, mean, std, coef, lower, upper):
    x = np.linspace(lower, upper, 300)
    plt.hist(sample, density=True, bins=50, facecolor="LightBlue", label="Frequency of X(T)")
    pdfvalue = pdfval(x, mean, std, coef)
    plt.plot(x, pdfvalue, color="r", label="density")
    plt.legend()
    plt.title("Histogram vs distribution")
    plt.xlabel("Y(T)")
    plt.show()

def pdfval(x, mean, std, coef):
    pdfvalue = coef[0] * ss.norm.pdf(x,loc=mean[0],scale=std[0])
    for i in range(1,len(mean)):
        pdfvalue += coef[i] * ss.norm.pdf(x,loc=mean[i],scale=std[i])
    return pdfvalue

def ou(x):
    return - kappa*x

def bsde(x, mean, std2):
    return kappa*x-sigma**2 * score(x, mean, std2)

def initdist(mu, sig2, coef, paths):
    u = ss.uniform.rvs(size=paths)
    
    X0 = np.zeros(paths)
    for i in range(paths):
        if u[i] <= coef[0]:
            X0[i] = ss.norm.rvs(loc=mu[0], scale=sig2[0])
        else:
            X0[i] = ss.norm.rvs(loc=mu[1], scale=sig2[1])
    return X0


X0 = initdist(mu, sig2, coef, paths)
X = euler(X0, ou, sigma, paths, T_vec)

N_processes = 10  # number of processes

# plotres(T_vec, X, 0, std_asy, -3, 3, N_processes)

plothist(X0, mu, sig2, coef, -4, 4)
plothist(X[:,-1], [0], [std_asy], [1], -4, 4)
# print(std_asy)
# print(ss.norm.fit(X[:,-1]))

Y0 = ss.norm.rvs(loc=0, scale=std_asy, size=paths)

WY = ss.norm.rvs(loc=0, scale=1, size=(paths, N - 1))
Y = np.zeros((paths, N))
Y[:,0] = Y0
# # Y_error = np.zeros((paths, N))
# # Y_error[:,0] = Y0
# t_lower = 10



mt, sigt2 = valuet(kappa, sigma, T_vec)
mean, std2 = distribution(mt, sigt2, mu, sig2)

for i in range(N-1):
    # mean, std = ss.norm.fit(X[:, i+1])
    scores = coef[0] * ss.norm.pdf(Y[:,i],loc=mean[0,i],scale=std2[0,i]) * score(Y[:,i], mean[0,i], std2[0,i])
    for j in range(1,len(mu)):
        scores += coef[j]*ss.norm.pdf(Y[:,i],loc=mean[j,i],scale=std2[j,i])*score(Y[:,i], mean[j,i], std2[j,i])
    density = pdfval(Y[:,i], mean[:,i], std2[:,i], coef)
    scores = scores/density
    Y[:, i+1] = Y[:,i] + (kappa*Y[:,i]+sigma**2 * scores)*dt + sigma*np.sqrt(dt)*WY[:,i]
    # Y_error[:, i+1] = Y_error[:,i] + (kappa*Y_error[:,i]-sigma**2 * score(Y)*dt + sigma*np.sqrt(dt)*WY[:,i]
    # std_vec[i] = 1/std2

plothist(Y[:,0], [0], [std_asy], [1], -4, 4)
plothist(Y[:,-1], mu, sig2, coef, -4, 4)

numbins=50
xhist = np.zeros((numbins, N))
yhist = np.zeros((numbins, N))
for i in range(N):
    xhist[:,i] = np.histogram(X[:,i],bins=numbins)[0]
    yhist[:,i] = np.histogram(Y[:,i],bins=numbins)[0]

fig = plt.figure(figsize=(16,5))
ax3 = fig.add_subplot(122)
x = np.linspace(-6, 6, 300)
ax3.hist(Y[:,-1], density=True, bins=50, facecolor="LightBlue", label="Frequency of Y(T)")
pdfvalue = pdfval(x, mu, sig2, coef)
ax3.plot(x, pdfvalue, color="r", label="density $X_0$")
ax3.set_title("Histogram vs distribution")
ax3.set_xlabel("Y(T)")
ax3.legend()
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(223)
y_vec = np.linspace(-6,6, numbins)
ax1.contourf(T_vec, y_vec, xhist, cmap=plt.get_cmap('Spectral'))
ax1.set_ylabel("density", rotation=0)
ax1.set_title("density ")
ax1.yaxis.set_label_coords(0,1.05)
ax2.contourf(T_vec, y_vec, yhist, cmap=plt.get_cmap('Spectral'))
ax2.invert_xaxis()
ax2.set_xlabel("time t")
plt.show()
