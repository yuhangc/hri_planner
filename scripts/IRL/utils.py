import theano as th
import theano.tensor as tt
import theano.tensor.slinalg as ts
import scipy.optimize
import numpy as np


def extract(var):
    return th.function([], var, mode=th.compile.Mode(linker='py'))()


def shape(var):
    return extract(var.shape)


def vector(n):
    return th.shared(np.zeros(n))


def matrix(n, m):
    return tt.shared(np.zeros((n, m)))


def grad(f, x, constants=[]):
    ret = th.gradient.grad(f, x, consider_constant=constants, disconnected_inputs='warn')
    if isinstance(ret, list):
        ret = tt.concatenate(ret)
    return ret


def jacobian(f, x):
    sz = shape(f)
    return tt.stacklists([grad(f[i], x) for i in range(sz)])


def hessian(f, x):
    return jacobian(grad(f, x), x)


class Minimizer(object):
    def __init__(self, f, vs, g={}, pre=None, gen=None, method='bfgs', eps=1, iters=100000, debug=False, inf_ignore=np.inf):
        self.inf_ignore = inf_ignore
        self.debug = debug
        self.iters = iters
        self.eps = eps
        self.method = method

        def one_gen():
            yield

        self.gen = gen
        if self.gen is None:
            self.gen = one_gen
        self.pre = pre
        self.f = f
        self.vs = vs
        self.sz = [shape(v)[0] for v in self.vs]
        for i in range(1,len(self.sz)):
            self.sz[i] += self.sz[i-1]
        self.sz = [(0 if i==0 else self.sz[i-1], self.sz[i]) for i in range(len(self.sz))]
        if isinstance(g, dict):
            self.df = tt.concatenate([g[v] if v in g else grad(f, v) for v in self.vs])
        else:
            self.df = g
        self.new_vs = [tt.vector() for v in self.vs]
        self.func = th.function(self.new_vs, [self.f, self.df], givens=zip(self.vs, self.new_vs))

        def f_and_df(x0):
            if self.debug:
                print x0
            s = None
            N = 0
            for _ in self.gen():
                if self.pre:
                    for v, (a, b) in zip(self.vs, self.sz):
                        v.set_value(x0[a:b])
                    self.pre()
                res = self.func(*[x0[a:b] for a, b in self.sz])
                if np.isnan(res[0]).any() or np.isnan(res[1]).any() or (np.abs(res[0])>self.inf_ignore).any() or (np.abs(res[1])>self.inf_ignore).any():
                    continue
                if s is None:
                    s = res
                    N = 1
                else:
                    s[0] += res[0]
                    s[1] += res[1]
                    N += 1
            s[0]/=N
            s[1]/=N
            return s
        self.f_and_df = f_and_df

    def argmin(self, vals={}, bounds={}):
        if not isinstance(bounds, dict):
            bounds = {v: bounds for v in self.vs}
        B = []
        for v, (a, b) in zip(self.vs, self.sz):
            if v in bounds:
                B += bounds[v]
            else:
                B += [(None, None)]*(b-a)
        x0 = np.hstack([np.asarray(vals[v]) if v in vals else v.get_value() for v in self.vs])
        if self.method=='bfgs':
            opt = scipy.optimize.fmin_l_bfgs_b(self.f_and_df, x0=x0, bounds=B)[0]
        elif self.method=='gd':
            opt = x0
            for _ in range(self.iters):
                opt -= self.f_and_df(opt)[1]*self.eps
        else:
            opt = scipy.optimize.minimize(self.f_and_df, x0=x0, method=self.method, jac=True).x
        return {v: opt[a:b] for v, (a, b) in zip(self.vs, self.sz)}

    def minimize(self, *args, **vargs):
        result = self.argmin(*args, **vargs)
        for v, res in result.iteritems():
            v.set_value(res)