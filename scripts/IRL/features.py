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


def goal_reward_term(x_goal):
    @feature
    def f(xh, uh):
        return tt.sum(tt.sqr(xh - x_goal))
    return f


def collision_hr(radius):
    @feature
    def f(xh, uh, xr, ur):
        return tt.exp(((xr[0] - xh[0])**2 + (xr[1] - xh[1])**2) / (radius**2))
    return f
