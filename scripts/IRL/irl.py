#!/usr/bin/env python

import numpy as np
# import features


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

    def likelihood_approx(self, th):
        """
        Requires compute_feature_grads to be called first
        """
        # compute g and H
        g = np.zeros_like(self.f_grad[0])
        H = np.zeros_like(self.f_hessian[0])

        for k in range(self.nfeature):
            g += th[k] * self.f_grad[k]
            H += th[k] * self.f_hessian[k]

        # compute the approximate log-likelihood
        h = np.linalg.solve(H, g)

        return 0.5 * (np.dot(g, h) + np.log(np.linalg.det(-H)) - self.du * np.log(2.0 * np.pi))

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
        h = Hinv * g

        grad = np.zeros_like(th)
        for k in range(self.nfeature):
            grad[k] = np.dot(h, self.f_grad[k]) - \
                      0.5 * np.dot(h, np.dot(self.f_hessian[k], h)) + \
                      0.5 * np.trace(Hinv * self.f_hessian[k])


class MaxEntIRL(object):
    def __init__(self, reward, *reward_args):
        """
        :param reward: the reward function class, used to initialize 
        :param features: 
        """
        self.reward = reward
        self.reward_args = reward_args
        self.n_demo = 0

        # data
        self.x = []
        self.u = []
        self.xr = []
        self.ur = []
        self.T = []

        # data dimensions
        self.nA = 0
        self.nX = 0
        self.nU = 0
        self.nXr = 0
        self.nUr = 0

    def load_data(self, data_path):
        # load the meta data info
        file_name = data_path + "/meta_data.txt"
        meta_data = np.loadtxt(file_name)

        self.n_demo = meta_data[0]
        self.nA = meta_data[1]
        self.nX = meta_data[2]
        self.nU = meta_data[3]
        self.nXr = meta_data[4]
        self.nUr = meta_data[5]

        # load data
        for d in range(self.n_demo):
            file_name = data_path + "/demo" + str(d) + ".txt"
            demo_data = np.loadtxt(file_name)
            col_start = 0

            # human poses
            self.x.append(demo_data[:, col_start:(self.nA*self.nX)])
            col_start += self.nA * self.nX

            self.u.append(demo_data[:, col_start:(col_start + self.nA*self.nU)])
            col_start += self.nA * self.nU

            self.xr.append(demo_data[:, col_start:(col_start + self.nXr)])
            col_start += self.nXr

            self.ur.append(demo_data[:, col_start:(col_start + self.nUr)])

            self.T.append(demo_data.shape[0])

    def optimize(self, th0, method="gd"):
        if method == "gd":
            self.optimize_gd(th0)
        elif method == "sgd":
            self.optimize_sgd(th0)
        else:
            raise Exception("Optimization method not recognized!")

    def optimize_gd(self, th0, n_iter=1000, lrate=0.05):
        # create a list of reward functions
        reward_func = []
        for d in range(self.n_demo):
            reward_func.append(self.reward(self.reward_args))
            # this only needs to be done once!
            reward_func[d].compute_feature_grads(self.x[d], self.u[d], self.xr[d], self.ur[d])

        # start iteration
        th = th0

        reward_hist = []
        for it in range(n_iter):
            # calculate (time-normalized) reward
            r = []
            for d in range(self.n_demo):
                r.append(reward_func[d].reward(th, self.x[d], self.u[d], self.xr[d], self.ur[d]) / self.T[d])

            # calculate gradient
            grad_th = np.zeros_like(th)
            for d in range(self.n_demo):
                grad_th += reward_func[d].likelihood_grad(th) / self.T[d]

            # gradient ascent
            th += lrate * grad_th

        return th

    def optimize_sgd(self, th0):
        pass
