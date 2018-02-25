#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

import dynamics
import features


class MaxEntIRLLinReward(object):
    def __init__(self, features):
        """
        :param features: list of features 
        """
        self.features = features
        self.nfeature = len(features)

        self.f_grad = None
        self.f_hessian = None

        self.du = 0

    def reward(self, th, x, u, xr, ur):
        r = 0.0

        # sum up all features
        for i in range(self.nfeature):
            r += th[i] * self.features[i](x, u, xr, ur)

        return r

    def compute_feature_grads(self, x, u, xr, ur):
        self.du = u.shape[1]

        self.f_grad = []
        self.f_hessian = []

        # compute for each feature
        for feature in self.features:
            self.f_grad.append(feature.grad(x, u, xr, ur))
            self.f_hessian.append(feature.hessian(x, u, xr, ur))

    def likelihood_approx_grad(self, th):
        # compute g and H
        g = np.zeros_like(self.f_grad[0])
        H = np.zeros_like(self.f_hessian[0])

        for k in range(self.nfeature):
            g += th[k] * self.f_grad[k]
            H += th[k] * self.f_hessian[k]

        # regularization
        H -= 0.1 * np.eye(H.shape[0])

        # compute H inverse
        Hinv = np.linalg.inv(H)

        # compute the gradient
        h = np.dot(Hinv, g)

        # approximation of likelihood
        Hdet = np.linalg.det(-H)
        if Hdet < 0:
            print "shouldn't happen"

        likelihood = 0.5 * (np.dot(g, h) + np.log(Hdet) - self.du * np.log(2.0 * np.pi))

        # gradient
        grad = np.zeros_like(th)
        for k in range(self.nfeature):
            aa = np.linalg.norm(h)
            bb = np.linalg.norm(self.f_grad[k])
            a = np.dot(h, self.f_grad[k])
            b = 0.5 * np.dot(h, np.dot(self.f_hessian[k], h))
            c = 0.5 * np.trace(np.dot(Hinv, self.f_hessian[k]))
            grad[k] = np.dot(h, self.f_grad[k]) - \
                      0.5 * np.dot(h, np.dot(self.f_hessian[k], h)) + \
                      0.5 * np.trace(np.dot(Hinv, self.f_hessian[k]))

        return likelihood, grad

    def likelihood_approx(self, th):
        """
        Requires compute_feature_grads to be called first
        """
        # compute g and H
        g = np.zeros_like(self.f_grad[0])
        H = np.zeros_like(self.f_hessian[0])

        for k in range(self.nfeature):
            g += th[k] * self.f_grad[k]
            # Hf = th[k] * self.f_hessian[k]
            # tmp = np.linalg.det(-Hf)
            # heigs = np.sort(np.linalg.eigvals(-Hf))
            H += th[k] * self.f_hessian[k]

        # compute the approximate log-likelihood
        h = np.linalg.solve(H, g)

        Hdet = np.linalg.det(-H)
        # heigs = np.sort(np.linalg.eigvals(-H))
        return 0.5 * (np.dot(g, h) + np.log(Hdet) - self.du * np.log(2.0 * np.pi))

    def likelihood_grad(self, th):
        """
        Requires compute_feature_grads to be called first
        """
        # compute g and H
        g = np.zeros_like(self.f_grad[0])
        H = np.zeros_like(self.f_hessian[0])

        for k in range(self.nfeature):
            g += th[k] * self.f_grad[k]
            H += th[k] * self.f_hessian[k]

        # compute H inverse
        Hinv = np.linalg.inv(H)

        # compute the gradient
        h = np.dot(Hinv, g)

        grad = np.zeros_like(th)
        for k in range(self.nfeature):
            # aa = np.dot(h, self.f_grad[k])
            # bb = -0.5 * np.dot(h, np.dot(self.f_hessian[k], h))
            # cc = 0.5 * np.trace(Hinv * self.f_hessian[k])
            # print aa, bb, cc
            grad[k] = np.dot(h, self.f_grad[k]) - \
                      0.5 * np.dot(h, np.dot(self.f_hessian[k], h)) + \
                      0.5 * np.trace(np.dot(Hinv, self.f_hessian[k]))
            # print k, ", ", grad[k]

        return grad


class MaxEntIRLBase(object):
    def __init__(self, xh, uh, xr, ur, meta_data):
        """
        :param xh: list of human trajectories
        :param uh: list of human controls
        :param xr: list of robot trajectories
        :param ur: list of robot controls
        :param T: trajectory horizon/length
        """
        # data
        self.xh = xh
        self.uh = uh
        self.xr = xr
        self.ur = ur

        self.dt = meta_data[0]
        self.T = meta_data[1]

        self.n_demo = len(self.xh)

        self.rewards = None

    def generate_rewards(self):
        raise Exception("Must be implemented by a derived class!")

    def optimize(self, th0, method="gd", n_iter=1000, lrate=0.05, verbose=False):
        if method == "gd":
            return self.optimize_gd(th0, n_iter, lrate, verbose)
        elif method == "sgd":
            return self.optimize_sgd(th0, n_iter, lrate, verbose)
        else:
            raise Exception("Optimization method not recognized!")

    def optimize_gd(self, th0, n_iter=1000, lrate=0.05, verbose=False):
        # initialize the reward gradients
        for d in range(self.n_demo):
            # this only needs to be done once!
            self.rewards[d].compute_feature_grads(self.xh[d], self.uh[d], self.xr[d], self.ur[d])

        # start iteration
        th = th0

        lhist = []
        for it in range(n_iter):
            log_l = 0.0
            grad_th = np.zeros_like(th)
            for d in range(self.n_demo):
                l, grad = self.rewards[d].likelihood_approx_grad(th)
                log_l += l
                grad_th += grad

            lhist.append(log_l)

            # gradient ascent
            grad_th /= self.n_demo
            th += lrate * grad_th

            # print out info
            if verbose and np.mod(it, 10) == 0:
                print "Iteration ", it, ", log likelihood is: ", log_l, \
                    ", gradient magnitude: ", np.linalg.norm(grad_th)

        return th, lhist

    def optimize_sgd(self, th0, n_iter=1000, lrate=0.05, verbose=False):
        pass


class HumanIRL(MaxEntIRLBase):
    def __init__(self, traj, x0, x_goal, obs, meta_data):
        xh, uh, xr, ur = traj
        super(HumanIRL, self).__init__(xh, uh, xr, ur, meta_data)

        self.x0 = x0
        self.x_goal = x_goal
        self.obs = obs

        # generate rewards
        self.rewards = self.generate_rewards()

    def generate_rewards(self):
        rlist = []

        for d in range(self.n_demo):
            # dynamics
            dyn = dynamics.ConstAccDynamics(self.dt)
            dyn.compute(self.x0[d], self.uh[d])

            # a list of features
            f_list = []

            # cumulative features
            # velocity
            f_vel = features.Velocity(dyn)
            f_list.append(f_vel)

            # acceleration
            f_acc = features.Acceleration(dyn)
            f_list.append(f_acc)

            # collision avoidance with robot
            # f_collision_hr = features.CollisionHRStatic(dyn, 0.3)
            f_collision_hr = features.CollisionHRStatic(dyn, offset=0.3)
            f_list.append(f_collision_hr)

            # dynamic collision avoidance with robot
            # f_collision_dyn = features.CollisionHRDynamic(dyn, 0.25, 0.3)
            f_collision_dyn = features.CollisionHRDynamic(dyn, 0.25, 0.25, offset=0.5)
            f_list.append(f_collision_dyn)

            # collision avoidance with static obstacle
            # f_collision_obs = features.CollisionObs(dyn, 0.3, self.obs[d])
            f_collision_obs = features.CollisionObs(dyn, self.obs[d], offset=0.3)
            f_list.append(f_collision_obs)

            # termination cost
            # goal
            f_goal = features.TerminationReward(dyn, self.x_goal[d])
            f_list.append(f_goal)

            # create the reward object
            rfunc = MaxEntIRLLinReward(f_list)
            rlist.append(rfunc)

        return rlist


def split_traj(x, T):
    T_total = len(x)
    x_out = []

    t = 0
    while (t+T) < T_total:
        x_out.append(x[t:(t+T)])
        t += 2

    # x_out.append(x[(T_total-T):T_total])

    return x_out


def load_data(path, n_training, T):
    file_name = path + "/init.txt"
    init_data = np.loadtxt(file_name, delimiter=',')

    file_name = path + "/goal.txt"
    goal_data = np.loadtxt(file_name, delimiter=',')

    file_name = path + "/obs.txt"
    obs_data = np.loadtxt(file_name, delimiter=',')

    xh_all = []
    uh_all = []
    xr_all = []
    ur_all = []
    x0_all = []
    xg_all = []
    obs = []

    for i in range(n_training):
        # initial condition
        x0 = init_data[i, 0:4]

        # load trajectory data
        file_name = path + "/block" + str(i) + ".txt"
        traj_data = np.loadtxt(file_name, delimiter=',')

        # split the data first
        traj_split = split_traj(traj_data, T)

        for traj_seg in traj_split:
            xh = traj_seg[:, 0:4]
            uh = traj_seg[:, 4:6]
            xr = traj_seg[:, 6:9]
            ur = traj_seg[:, 9:11]

            # split the trajectory and add to data point
            xh_all.append(xh)
            uh_all.append(uh)
            xr_all.append(xr)
            ur_all.append(ur)

            # initial condition
            x0_all.append(x0)
            x0 = xh[T-1]

            # goal
            xg_all.append(goal_data[i])

            # obstacle
            obs.append(obs_data)

    return xh_all, uh_all, xr_all, ur_all, x0_all, xg_all, obs


if __name__ == "__main__":
    # load data
    n_users = 4
    n_user_demo = [40, 20, 20, 30]
    T = 10
    cond = "rp"

    xh = []
    uh = []
    xr = []
    ur = []
    x0 = []
    xg = []
    obs = []

    for i in range(n_users):
        path = "/home/yuhang/Documents/irl_data/winter18/user" + str(i) + "/processed/" + cond
        xhi, uhi, xri, uri, x0i, xgi, obsi = load_data(path, n_user_demo[i], T)

        xh += xhi
        uh += uhi
        xr += xri
        ur += uri
        x0 += x0i
        xg += xgi
        obs += obsi

    # create IRL object
    irl = HumanIRL((xh, uh, xr, ur), x0, xg, obs, [0.5, T])

    # optimize
    # th0 = np.array([-7, -20.0, -0.1, -0.1, -0.1, -35.])
    # th0 = np.array([-7, -20.0, -1.5, -1.5, -35.])
    th0 = np.array([-8., -27.0, -3.6, -1.0, -1.5, -41.])
    th_opt, lhist = irl.optimize(th0, n_iter=100, verbose=True)

    print th_opt

    plt.plot(lhist, '-b')
    plt.show()
