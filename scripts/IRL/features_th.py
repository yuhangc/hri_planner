#!/usr/bin/env python

import theano as th
import theano.tensor as tt
import numpy as np

import utils
import dynamics_th
import dynamics
import features
from irl import load_data


class Feature(object):
    def __init__(self, f):
        self.f = f

    def __call__(self, *args):
        return self.f(*args)

    def __add__(self, r):
        return Feature(lambda *args: self(*args)+r(*args))

    def __radd__(self, r):
        return Feature(lambda *args: r(*args)+self(*args))

    def __mul__(self, r):
        return Feature(lambda *args: self(*args)*r)

    def __rmul__(self, r):
        return Feature(lambda *args: r*self(*args))

    def __pos__(self, r):
        return self

    def __neg__(self):
        return Feature(lambda *args: -self(*args))

    def __sub__(self, r):
        return Feature(lambda *args: self(*args)-r(*args))

    def __rsub__(self, r):
        return Feature(lambda *args: r(*args)-self(*args))


def feature(f):
    return Feature(f)


# ----------------------------- cumulative costs -----------------------------------
def velocity():
    @feature
    def f(xh, uh, xr, ur):
        return xh[2]**2 + xh[3]**2
    return f


def acceleration():
    @feature
    def f(xh, uh, xr, ur):
        return uh[0]**2 + uh[1]**2
    return f


def collision_hr(radius):
    @feature
    def f(xh, uh, xr, ur):
        return tt.exp(-((xr[0] - xh[0])**2 + (xr[1] - xh[1])**2) / (radius**2))
    return f


def collision_hr_dynamic(w, l, dt):
    @feature
    def f(xh, uh, xr, ur):
        # compute center
        th = xr[2]
        xc = xr[0] + ur[0] * dt * tt.cos(th)
        yc = xr[1] + ur[0] * dt * tt.sin(th)

        # compute Gaussian length and width
        gw = w
        gl = l + ur[0] * 2.0 * l

        # convert to robot reference frame
        d = (xh[0] - xc, xh[1] - yc)

        x_hr = tt.cos(th) * d[0] + tt.sin(th) * d[1]
        y_hr = -tt.sin(th) * d[0] + tt.cos(th) * d[1]

        # compute cost
        return tt.exp(-(x_hr**2/(gw**2) + y_hr**2/(gl**2)))
    return f


def collision_obs(radius, pos):
    @feature
    def f(xh, uh, xr, ur):
        return tt.exp(-((pos[0] - xh[0])**2 + (pos[1] - xh[1])**2) / (radius**2))
    return f


# ----------------------------- termination costs -----------------------------------
def goal_reward_term(x_goal):
    @feature
    def f(xh, uh):
        return tt.sqrt(tt.sum(tt.sqr(xh - x_goal)))
    return f


# ----------------------------- validation -----------------------------------
def feature_grad_validation(xhi, uhi, xri, uri, x0i, xgi, obs):
    dt = 0.5
    T = 10

    nX = 4
    nU = 2
    nXr = 3
    nUr = 2

    # initial/goal position
    x0 = utils.vector(nX)
    x_goal = utils.vector(nX)

    # robot data
    xr = [utils.vector(nXr) for t in range(T)]
    ur = [utils.vector(nUr) for t in range(T)]

    # generate human trajectory function
    uh = [utils.vector(nU) for t in range(T)]

    # dynamics
    dyn = dynamics_th.DynamicsConstAacc()

    # generate the trajectory function
    xh = []

    x_next = x0
    for t in range(T):
        x_next = dyn(x_next, uh[t], dt)
        xh.append(x_next)

    # generate theano features
    f_th = []
    f_th.append(velocity())
    f_th.append(acceleration())
    f_th.append(collision_hr(0.3))
    f_th.append(collision_hr_dynamic(0.25, 0.3, 0.5))
    f_th.append(collision_obs(0.5, obs))
    # f_th.append(goal_reward_term(x_goal))

    # set values
    for u, uval in zip(uh, uhi):
        u.set_value(uval)
    for x, xval in zip(xr, xri):
        x.set_value(xval)
    for u, uval in zip(ur, uri):
        u.set_value(uval)
    x0.set_value(x0i)
    x_goal.set_value(xgi)

    # generate normal features
    # dynamics
    dyn = dynamics.ConstAccDynamics(dt)
    dyn.compute(x0i, uhi)

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
    f_collision_hr = features.CollisionHRStatic(dyn, 0.3)
    f_list.append(f_collision_hr)

    # dynamic collision avoidance with robot
    f_collision_dyn = features.CollisionHRDynamic(dyn, 0.25, 0.3)
    f_list.append(f_collision_dyn)

    # collision avoidance with static obstacle
    f_collision_obs = features.CollisionObs(dyn, 0.5, obs)
    f_list.append(f_collision_obs)

    # termination cost
    # goal
    # f_goal = features.TerminationReward(dyn, xgi)
    # f_list.append(f_goal)

    # compare
    fid = 0
    for f, ft in zip(f_list, f_th):
        grad = f.grad(xhi, uhi, xri, uri)

        # theano gradient
        fsum = sum([ft(xh[t], uh[t], xr[t], ur[t]) for t in range(T)])
        gth = utils.grad(fsum, uh)

        new_vs = [tt.vector() for v in uh]
        gfunc = th.function(new_vs, [fsum, gth], givens=zip(uh, new_vs))

        sz = [utils.shape(v)[0] for v in uh]
        for i in range(1, len(sz)):
            sz[i] += sz[i-1]
        sz = [(0 if i==0 else sz[i-1], sz[i]) for i in range(len(sz))]

        uhv = uhi.flatten()
        val = gfunc(*[uhv[a:b] for a, b in sz])

        fval = val[0]
        gval = val[1]

        f_diff = f(xhi, uhi, xri, uri) - fval
        g_diff = grad.flatten() - gval
        print "difference for feature ", fid, " is ", f_diff
        print g_diff
        assert np.sum(np.abs(g_diff)) < 1e-6

        fid += 1


if __name__ == "__main__":
    path = "/home/yuhang/Documents/irl_data/winter18/user0/processed/hp"
    xhi, uhi, xri, uri, x0i, xgi, obsi = load_data(path, 5, 10)

    feature_grad_validation(xhi[0], uhi[0], xri[0], uri[0], x0i[0], xgi[0], obsi[0])
