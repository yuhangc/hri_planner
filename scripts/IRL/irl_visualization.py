#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator

import features_th as features


# redefine the features to fit in the visualization framework
def collision_hr(xr, radius):
    @features.feature
    def f(xh):
        return np.exp(-((xr[0] - xh[0])**2 + (xr[1] - xh[1])**2) / (radius**2))
    return f


def collision_hr_dynamic(xr, ur, w, l, dt=0.5):
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
        return np.exp(-(x_hr**2/(gl**2) + y_hr**2/(gw**2)))
    return f


def collision_obs(pos, radius):
    @features.feature
    def f(xh):
        return np.exp(-((pos[0] - xh[0])**2 + (pos[1] - xh[1])**2) / (radius**2))
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
            a = f(np.array([x[i, j], y[i, j]]))
            z[i, j] = f(np.array([x[i, j], y[i, j]]))

    # set levels and color map
    levels = MaxNLocator(nbins=30).tick_values(z.min(), z.max())
    cmap = plt.get_cmap("Spectral")
    # norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    cf = ax.contourf(x, y, z, levels=levels, cmap=cmap)
    fig.colorbar(cf, ax=ax)

    # ax.axis("equal")


def visualize_features_basic():
    # create the features
    xr = np.array([2.0, 2.0, np.pi / 6.0])
    ur = np.array([0.5, 0.0])
    obs_pos = np.array([2.0, 3.0])
    x_goal = np.array([0.5, 6.0])

    f_chr = -collision_hr(xr, 0.5)
    f_chr_dyn = -collision_hr_dynamic(xr, ur, 0.5, 0.5, dt=1.0)
    f_obs = -collision_obs(obs_pos, 0.5)
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
    obs_pos = np.loadtxt(path + "/obs.txt", delimiter=',')

    traj = np.loadtxt(path + "/block" + str(trial) + ".txt", delimiter=",")

    xh = traj[1:17, 0:4]
    uh = traj[1:17, 4:6]
    xr = traj[1:17, 6:9]
    ur = traj[1:17, 9:11]

    # extract common info
    x0 = init_data[trial, 0:4]
    x_goal = goal_data[trial]

    # bound
    xbound = np.array([-1.0, 5.0])
    ybound = np.array([0.0, 7.0])

    # loop though the time steps
    fig, axes = plt.subplots(2, 4)

    nstep = 2
    for t in range(0, len(xr), nstep):
        # generate features
        f_chr = -collision_hr(xr[t], 0.5)
        f_chr_dyn = -collision_hr_dynamic(xr[t], ur[t], 0.3, 0.5, dt=1.0)
        f_obs = -collision_obs(obs_pos, 0.5)
        f_goal = -goal_reward_term(x_goal)

        f_all = th[0]*f_chr + th[1]*f_chr_dyn + th[2]*f_obs + th[3]*f_goal
        ax = axes[(t/2) / 4, (t/2) % 4]
        plot_cost_heat_map(f_all, xbound, ybound, fig, ax)

        # overlay the partial trajectory
        if t > 1:
            ax.plot(xh[:t-1, 0], xh[:t-1, 1], '-', color="grey", lw=1.5, label="human")
            ax.plot(xr[:t-1, 0], xr[:t-1, 1], '-', color="grey", lw=1.5, label="robot")
            ax.plot(xh[t-2:t+1, 0], xh[t-2:t+1, 1], '-ok', lw=1.5, fillstyle="none", label="human")
            ax.plot(xr[t-2:t+1, 0], xr[t-2:t+1, 1], '-or', lw=1.5, fillstyle="none", label="robot")
        else:
            ax.plot(xh[:t, 0], xh[:t, 1], '-k', lw=1.5, label="human")
            ax.plot(xr[:t, 0], xr[:t, 1], '-r', lw=1.5, label="robot")

    plt.show()


if __name__ == "__main__":
    # visualize_features_basic()
    visualize_features_with_data("/home/yuhang/Documents/irl_data/winter18/user0/processed/rp",
                                 0, [7.0, 10.0, 8.0, 50.0/100.0])
