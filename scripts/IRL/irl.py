#!/usr/bin/env python

import numpy as np
# import features as irl_features


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
                      0.5 * np.trace(Hinv * self.f_hessian[k])
            # print k, ", ", grad[k]

        return grad


class MaxEntIRL(object):
    def __init__(self, initializer):
        """
        :param initializer: an initializer that loads data and generate reward functions
        """
        # initializer
        self.initializer = initializer

        self.n_demo = 0

        # data
        self.x = []
        self.u = []
        self.xr = []
        self.ur = []
        self.T = []

        # rewards
        self.rewards = None

    def init(self, data_path):
        # load data
        self.x, self.u, self.xr, self.ur, self.T = self.initializer.load_data(data_path)
        self.n_demo = len(self.x)

        # generate reward functions
        self.rewards = self.initializer.generate_rewards()

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
            self.rewards[d].compute_feature_grads(self.x[d], self.u[d], self.xr[d], self.ur[d])

        # start iteration
        th = th0

        lhist = []
        for it in range(n_iter):
            # calculate (time-normalized) reward
            # r = 0.0
            # for d in range(self.n_demo):
            #     r += self.rewards[d].reward(th, self.x[d], self.u[d], self.xr[d], self.ur[d]) / self.T[d]
            # reward_hist.append(r)
            log_l = 0.0
            for d in range(self.n_demo):
                log_l += self.rewards[d].likelihood_approx(th)
            lhist.append(log_l)

            # calculate gradient
            grad_th = np.zeros_like(th)
            for d in range(self.n_demo):
                grad_th += self.rewards[d].likelihood_grad(th) / self.T[d]

            # gradient ascent
            th += lrate * grad_th / self.n_demo

            # print out info
            if verbose and np.mod(it, 10) == 0:
                print "Iteration ", it, ", log likelihood is: ", log_l, \
                    ", gradient magnitude: ", np.linalg.norm(grad_th)

        return th, lhist

    def optimize_sgd(self, th0, n_iter=1000, lrate=0.05, verbose=False):
        pass
