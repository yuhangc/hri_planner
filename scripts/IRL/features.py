#!/usr/bin/env python

import theano as th
import theano.tensor as tt


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
        return tt.exp(-0.5 * (x_hr**2/(gw**2) + y_hr**2/(gl**2)))
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
        return tt.sum(tt.sqr(xh - x_goal))
    return f
