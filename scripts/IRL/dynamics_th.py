#!/usr/bin/env python

import numpy as np
import theano.tensor as tt


class DynamicsBase(object):
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        return self.forward_dyn_th(*args)

    def forward_dyn_th(self, *args):
        raise Exception("Must be implemented by a derived class!")

    def forward_dyn_np(self, *args):
        raise Exception("Must be implemented by a derived class!")


class DynamicsConstAacc(DynamicsBase):
    def forward_dyn_th(self, x_prev, u, dt):
        return tt.stacklists([
            x_prev[0] + x_prev[2] * dt + 0.5 * u[0] * dt**2,
            x_prev[1] + x_prev[3] * dt + 0.5 * u[1] * dt**2,
            x_prev[2] + u[0] * dt,
            x_prev[3] + u[1] * dt
        ])

    def forward_dyn_np(self, x_prev, u, dt):
        return np.array([
            x_prev[0] + x_prev[2] * dt + 0.5 * u[0] * dt**2,
            x_prev[1] + x_prev[3] * dt + 0.5 * u[1] * dt**2,
            x_prev[2] + u[0] * dt,
            x_prev[3] + u[1] * dt
        ])
