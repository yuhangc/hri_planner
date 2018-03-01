#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator

import features_th as features


# redefine the features to fit in the visualization framework
def collision_hr(xr, offset):
    @features.feature
    def f(xh):
        return 1.0 / ((xh[0] - xr[0])**2 + (xh[1] - xr[1])**2 + offset)
    return f


def collision_hr_dynamic(xr, ur, w, l, offset, dt=0.5):
    @features.feature
    def f(xh):
        # compute center
        th = xr[2]
        xc = xr[0] + ur[0] * dt * np.cos(th)
        yc = xr[1] + ur[0] * dt * np.sin(th)

        # compute Gaussian length and width
        gw = w
        gl = l + ur[0] * 2.0 * l

        # convert to robot reference frame
        d = (xh[0] - xc, xh[1] - yc)

        x_hr = np.cos(th) * d[0] + np.sin(th) * d[1]
        y_hr = -np.sin(th) * d[0] + np.cos(th) * d[1]

        # compute cost
        return 1.0 / (x_hr**2/(gl**2) + y_hr**2/(gw**2) + offset)
    return f


def collision_obs(pos, offset):
    @features.feature
    def f(xh):
        return 1.0 / ((pos[0] - xh[0])**2 + (pos[1] - xh[1])**2 + offset)
    return f


def goal_reward_term(x_goal):
    @features.feature
    def f(xh):
        return np.linalg.norm(xh - x_goal)
    return f


def plot_cost_heat_map(f, xbound, ybound, fig, ax):
    # generate mesh
    nres = 100
    xx = np.linspace(xbound[0], xbound[1], nres)
    yy = np.linspace(ybound[0], ybound[1], nres)

    x, y = np.meshgrid(xx, yy)

    # compute the feature values
    z = np.zeros_like(x)

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            z[i, j] = f(np.array([x[i, j], y[i, j]]))

    # set levels and color map
    levels = MaxNLocator(nbins=30).tick_values(z.min(), z.max())
    cmap = plt.get_cmap("Spectral")
    # norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    cf = ax.contourf(x, y, z, levels=levels, cmap=cmap)
    fig.colorbar(cf, ax=ax)

    ax.axis("equal")


def visualize_features_basic():
    # create the features
    xr = np.array([2.0, 2.0, np.pi / 6.0])
    ur = np.array([0.5, 0.0])
    obs_pos = np.array([2.0, 3.0])
    x_goal = np.array([0.5, 6.0])

    f_chr = -collision_hr(xr, 0.1)
    f_chr_dyn = -collision_hr_dynamic(xr, ur, 1.0, 1.0, 0.1, dt=1.5)
    f_obs = -collision_obs(obs_pos, 0.1)
    f_goal = -goal_reward_term(x_goal)

    # bound
    xbound = np.array([0.0, 6.0])
    ybound = np.array([0.0, 6.0])

    # visualize each feature individually
    fig, axes = plt.subplots(2, 2)
    plot_cost_heat_map(f_chr, xbound, ybound, fig, axes[0, 0])
    plot_cost_heat_map(f_chr_dyn, xbound, ybound, fig, axes[0, 1])
    plot_cost_heat_map(f_obs, xbound, ybound, fig, axes[1, 0])
    plot_cost_heat_map(f_goal, xbound, ybound, fig, axes[1, 1])
    plt.show()

    # visualize sum of the features
    th = [1.5, 1.0, 1.2, 45.0/100.0]
    fig, axes = plt.subplots()
    plot_cost_heat_map(th[0]*f_chr + th[1]*f_chr_dyn + th[2]*f_obs + th[3]*f_goal,
                       xbound, ybound, fig, axes)
    plt.show()


def visualize_features_with_data(path, trial, th):
    # load the data
    init_data = np.loadtxt(path + "/init.txt", delimiter=",")
    goal_data = np.loadtxt(path + "/goal.txt", delimiter=",")
    obs_data = np.loadtxt(path + "/obs.txt", delimiter=',')

    traj = np.loadtxt(path + "/block" + str(trial) + ".txt", delimiter=",")

    xh = traj[:, 0:4]
    uh = traj[:, 4:6]
    xr = traj[:, 6:9]
    ur = traj[:, 9:11]

    # extract common info
    x0 = init_data[trial, 0:4]
    x_goal = np.zeros_like(x0)
    x_goal[0:2] = goal_data[trial]


if __name__ == "__main__":
    visualize_features_basic()
