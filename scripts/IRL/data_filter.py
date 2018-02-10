import numpy as np
import theano as th
import theano.tensor as tt
import matplotlib.pyplot as plt

import utils


class OptimalFilter(object):
    def __init__(self, dt, w, T):
        self.dt = dt
        self.w = w

        self.nX = 4
        self.nU = 2

        self.T = T
        self.x = [utils.vector(self.nX) for t in range(self.T)]
        self.u = [utils.vector(self.nU) for t in range(self.T)]

        self.x0 = utils.vector(self.nX)
        self.x_goal = utils.vector(self.nX)

        self.x_pred = None
        self.gen()

        self.optimizer = utils.Minimizer(self.cost(), self.u)

    def gen(self):
        # generate the trajectory function
        self.x_pred = []

        x_next = self.x0
        for t in range(self.T):
            x_next = self.forward_dynamics(x_next, self.u[t])
            self.x_pred.append(x_next)

    def set_end_points(self, x0, x_goal):
        self.x0.set_value(np.asarray(x0))
        self.x_goal.set_value(np.asarray(x_goal))

    def set_traj(self, x):
        for t in range(self.T):
            self.x[t].set_value(x[t])

    def cost(self):
        costs = [self.w[0] * self.cost_pos_match(self.x[t], self.x_pred[t]) +
                 self.w[1] * self.cost_vel(self.x_pred[t]) +
                 self.w[2] * self.cost_acc(self.u[t]) for t in range(self.T-1)]

        return sum(costs) + self.w[3] * self.cost_goal_match(self.x_goal, self.x_pred[self.T-1])

    def filter_data(self, x):
        # set trajectory and generate functions
        self.set_traj(x)

        # TODO: set initial values

        # prepare to optimize
        self.optimizer.minimize()

        # return optimal trajectory and control
        return self.get_traj_control()

    def get_traj_control(self):
        xout = []
        uout = []

        x_val = self.x0.get_value()
        for t in range(self.T):
            u_val = self.u[t].get_value()
            x_val = self.forward_dynamics_np(x_val, u_val)
            xout.append(x_val)
            uout.append(u_val)

        return xout, uout

    def forward_dynamics(self, x_prev, u):
        return tt.stacklists([
            x_prev[0] + x_prev[2] * self.dt + 0.5 * u[0] * self.dt**2,
            x_prev[1] + x_prev[3] * self.dt + 0.5 * u[1] * self.dt**2,
            x_prev[2] + u[0] * self.dt,
            x_prev[3] + u[1] * self.dt
        ])

    def forward_dynamics_np(self, x_prev, u):
        return np.array([
            x_prev[0] + x_prev[2] * self.dt + 0.5 * u[0] * self.dt**2,
            x_prev[1] + x_prev[3] * self.dt + 0.5 * u[1] * self.dt**2,
            x_prev[2] + u[0] * self.dt,
            x_prev[3] + u[1] * self.dt
        ])

    @staticmethod
    def cost_pos_match(x, x_pred):
        return tt.sum(tt.sqr(x[0:2] - x_pred[0:2]))

    @staticmethod
    def cost_goal_match(x_goal, x_pred):
        return tt.sum(tt.sqr(x_goal - x_pred))

    @staticmethod
    def cost_vel(x_pred):
        return tt.sum(tt.sqr(x_pred[2:4]))

    @staticmethod
    def cost_acc(u):
        return tt.sum(tt.sqr(u))


if __name__ == "__main__":
    # create a filter object
    traj_filter = OptimalFilter(0.5, [1.0, 0.1, 0.1, 1.0], 16)

    # set start position and goal
    traj_filter.set_end_points([3.22, 0.84, 0.0, 0.0], [0.73, 6.05, 0.0, 0.0])
    
    # set test trajectory
    x = np.array([[ 3.13019,  9.68101e-01, -2.44455e-01,  4.15714e-01],
                 [ 2.96550,  1.27273, -2.56095e-01,  6.50851e-01],
                [ 3.02913,  1.64058,  3.71875e-01,  7.27233e-01],
                [ 2.99557,  2.06195, -3.08982e-01,  8.30587e-01],
                [ 2.88412,  2.49200, -1.04329e-01,  8.73822e-01],
                [ 2.87438,  2.93936, -1.84359e-01,  8.90539e-01],
                [ 2.58098,  3.36864, -6.15615e-01,  8.75663e-01],
                [ 2.38843,  3.85366, -2.79745e-01,  1.00373],
                [ 2.06469,  4.27498, -8.23626e-01,  7.78557e-01],
                [ 1.75451,  4.69639, -4.85934e-01,  8.85798e-01],
                [ 1.54210,  5.14777, -5.49390e-01,  7.83216e-01],
                [ 1.19693,  5.42890, -6.29426e-01,  6.39690e-01],
                [ 1.00202,  5.75015, -3.27712e-01,  7.79959e-01],
                [ 8.39313e-01,  5.97561, -2.99222e-01,  1.89099e-01],
                [ 7.47053e-01,  6.03368, -1.41895e-01,  1.64901e-01],
                [ 7.27136e-01,  6.04965, -2.96544e-02, -4.57665e-03]])

    x_filtered, u = traj_filter.filter_data(x)
    x_plt = np.asarray(x_filtered)
    u_plt = np.asarray(u)

    # plot
    fig, axes = plt.subplots(3, 1)

    axes[0].plot(x_plt[:, 0], '-k')
    axes[0].plot(x_plt[:, 1], '--k')
    axes[0].plot(x[:, 0], '-b')
    axes[0].plot(x[:, 1], '--b')

    axes[1].plot(x_plt[:, 2], '-k')
    axes[1].plot(x_plt[:, 3], '--k')
    axes[1].plot(x[:, 2], '-b')
    axes[1].plot(x[:, 3], '--b')

    axes[2].plot(u_plt[:, 0], '-k')
    axes[2].plot(u_plt[:, 1], '--k')

    fig, axes = plt.subplots()
    axes.plot(x_plt[:, 0], x_plt[:, 1], '-k', lw=2, marker='o', markersize=10, fillstyle="none")
    axes.plot(x[:, 0], x[:, 1], '-b', lw=2, marker='o', markersize=10, fillstyle="none")

    plt.show()

    print "first optimization finished"

    # change x a little
    traj_filter.set_end_points([4.22, 1.84, 0.0, 0.0], [1.73, 7.05, 0.0, 0.0])

    x[:, 0:2] += 1.0
    x_filtered, u = traj_filter.filter_data(x)
    x_plt = np.asarray(x_filtered)
    u_plt = np.asarray(u)

    # plot
    fig, axes = plt.subplots(3, 1)

    axes[0].plot(x_plt[:, 0], '-k')
    axes[0].plot(x_plt[:, 1], '--k')
    axes[0].plot(x[:, 0], '-b')
    axes[0].plot(x[:, 1], '--b')

    axes[1].plot(x_plt[:, 2], '-k')
    axes[1].plot(x_plt[:, 3], '--k')
    axes[1].plot(x[:, 2], '-b')
    axes[1].plot(x[:, 3], '--b')

    axes[2].plot(u_plt[:, 0], '-k')
    axes[2].plot(u_plt[:, 1], '--k')

    fig, axes = plt.subplots()
    axes.plot(x_plt[:, 0], x_plt[:, 1], '-k', lw=2, marker='o', markersize=10, fillstyle="none")
    axes.plot(x[:, 0], x[:, 1], '-b', lw=2, marker='o', markersize=10, fillstyle="none")

    plt.show()

    print "second optimization finished"
