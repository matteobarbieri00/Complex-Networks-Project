import numpy as np
from random import *
import matplotlib.pyplot as plt
import scipy
from scipy.optimize import curve_fit


# function that implements a classical random walk on a line
def c_rw_1D(lenght, Delta_x, Delta_t, time, x_0, pl):
    x = x_0
    T = int(time / Delta_t)
    for i in range(T):
        if x != (-lenght) and x != lenght:
            p = random()
            if p <= pl:
                x = x - Delta_x
            else:
                x = x + Delta_x
        elif x == (-lenght):
            p = random()
            if p <= pl:
                x = lenght
            else:
                x = x + Delta_x
        elif x == lenght:
            p = random()
            if p <= pl:
                x = x - Delta_x
            else:
                x = -lenght
    return x


# implementing the line and doing the random walk for pr=pl=0.5
L = 100
Delta_x = 1
Delta_t = 1
T = 100
x_0 = 0
positions = []
particles = 10000
for i in range(particles):
    positions.append(c_rw_1D(L, Delta_x, Delta_t, T, x_0, 0.5))

bins = 2 * L
x = []
y = []
n_in_pos = [0 for i in range(-L, L + 1)]
for i in range(-L, L + 1):
    for j in range(len(positions)):
        if positions[j] == i:
            n_in_pos[i] += 1

for i in range(-L, L + 1):
    if i % 2 == 0:
        x.append(i)
        y.append(n_in_pos[i] / particles)


# Recast xdata and ydata into numpy arrays so we can use their handy features
xdata = np.asarray(x)
ydata = np.asarray(y)
# plt.plot(xdata, ydata, "o")


n = len(xdata)
mean = sum(xdata * ydata) / n
sigma = sum(ydata * (xdata - mean) ** 2) / n


def gaus(x, a, x0, sigma):
    return a * np.exp(-((x - x0) ** 2) / (2 * sigma**2))


popt, pcov = curve_fit(gaus, x, y, p0=[1, mean, sigma])

fig, ax = plt.subplots()
ax.plot(x, y, "b+:", label="data")
ax.plot(x, gaus(x, *popt), "ro:", label="fit")
plt.legend()
# plt.title('Fig. 3 - Fit for Time Constant')
plt.xlabel("Position")
plt.ylabel("Probability")
textstr = "\n".join((r"$\mu=%.2f$" % (popt[1],), r"$\sigma=%.2f$" % (popt[2],)))
# these are matplotlib.patch.Patch properties
props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)

# place a text box in upper left in axes coords
ax.text(
    0.05,
    0.95,
    textstr,
    transform=ax.transAxes,
    fontsize=14,
    verticalalignment="top",
    bbox=props,
)
plt.show()
