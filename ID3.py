from __future__ import annotations
from typing import Dict
from collections import Counter
from math import log
from itertools import groupby
import numpy as np
import pandas as pd
from copy import deepcopy


class ID3:
    def __init__(
        self,
        attributes_size: int,
        X: np.array,
        Y: np.array
    ):
        self.attributes_size = attributes_size
        self.attributes = list(range(attributes_size))
        self.XY = np.concatenate((X, Y), axis=1).astype(str)

        self.classes = np.unique(Y[:, 0]).astype(str)  # unique values of Y
        self.tree = None

    def groupby(self, XY: np.array, indx: int):
        groups = {}
        for row in XY:
            if row[indx] not in groups.keys():
                groups[row[indx]] = [row]
            else:
                groups[row[indx]].append(row)
        return [np.array(a) for a in groups.values()]

    def countEntropy(self, XY):
        groups = self.groupby(XY, -1)
        entropy = 0
        for group in groups:
            size = len(group)/XY[:, 0].size
            entropy -= size * log(size)
        return entropy

    def countInfGain(self, X, Y):
        size = Y.size
        XY = np.concatenate([X.reshape(size, 1), Y.reshape(size, 1)], axis=1)
        groups = self.groupby(XY, 0)
        inf_gain = self.countEntropy(XY)
        for group in groups:
            inf_gain -= (
                    group[:, 0].size / size *
                    self.countEntropy(group)
                )
        return inf_gain

    def find_inf_gain_maximizing(self, attributes, XY) -> int:  # returns the index of column that should by used
        gains = []
        for i in attributes:
            gains.append(self.countInfGain(XY[:, i], XY[:, -1]))
        results = list(zip(gains, attributes))
        return max(results, key=lambda result: result[0])[1]

    def learn(self) -> Dict:
        # create copy of parameters
        XY = deepcopy(self.XY)
        attributes = self.attributes
        self.tree = self.learn_rec(attributes, XY)

    def learn_rec(self, attributes, XY) -> Dict:
        if XY.size == 0:
            raise ValueError  # no elements in X
        temp = XY[:, -1]
        if (temp[0] == temp).all():
            return temp[0]  # all elements of Y are the same
        if attributes == []:
            values, counts = np.unique(XY[:, -1], return_counts=True)
            return values[np.argmax(counts)]  # return element that occurs the most
        D = self.find_inf_gain_maximizing(attributes, XY)  # int
        D_values = np.unique(XY[:, D])  # possible values in D column
        result = {D: {}}
        attributes.remove(D)
        for D_value in D_values:
            subset = XY[XY[:, D] == D_value, :]
            result[D][D_value] = self.learn_rec(attributes, subset)
        return result

    def predict(self, sample: np.array):
        sample = sample.astype(str)
        if not self.tree:
            raise ValueError
        if type(self.tree) is not type({}):
            return self.tree
        current_index = 0
        current_node = self.tree
        while True:
            current_index = list(current_node.keys())[0]
            edges = current_node[current_index]
            current_node = edges[sample[current_index]]
            if type(current_node) is not type({}):
                return current_node

