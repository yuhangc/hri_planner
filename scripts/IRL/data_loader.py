#!/usr/bin/env python

import numpy as np


class DataLoader(object):
    def __init__(self):
        # data dimensions
        self.dt = 0.0
        self.nA = 0
        self.nX = 0
        self.nU = 0
        self.nXr = 0
        self.nUr = 0

        # store the data in variables
        self.x_raw = []
        self.u = []

        # meta data
        self.meta_data = None