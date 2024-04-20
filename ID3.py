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
        X: pd.Dataframe,
        Y: pd.Dataframe
    ):
        self.atttributes_size = attributes_size
        self.attributes = list(range(attributes_size))
        self.X = X
        self.Y = Y
        self.classes = pd.unique(Y.iloc[:, 0])  # unique values of Y
        self.tree = None

    def countEntropy(self, XY):
        groups = list(XY.groupby(XY.columns[1]))
        entropy = 0
        for _, group in groups:
            size = group[group.columns[0]].size
            entropy -= size * log(size)
        return entropy

    def countInfGain(self, X, Y):
        size = Y.size
        pairs = pd.concat([X, Y], axis=1)
        groups = list(pairs.groupby(pairs.columns[0]))
        inf_gain = 0
        for _, group in groups:
            inf_gain += (
                group[group.columns[0]].size / size *
                self.countEntropy(group[group.columns])
                )
        return inf_gain

    def find_inf_gain_maximizing(self, attributes, X, Y) -> int:  #returns the index of column that should by used
        gains = []
        for i in attributes:
            gains.append(self.countInfGain(X.iloc[:, i], Y.iloc[:, 0]))
        results = list(zip(gains, attributes))
        return max(results, key=lambda result: result[0])[1]

    def learn(self) -> Dict:
        # create copy of parameters
        X = deepcopy(self.X)
        Y = deepcopy(self.Y)
        attributes = self.attributes
        self.tree = self.learn_rec(attributes, X, Y)
        print(self.tree)

    def learn_rec(self, attributes, X, Y) -> Dict:
        if X.size is None:
            raise ValueError  # no elements in X
        if all(y == Y.iloc[0, 0] for y in Y.iloc[:, 0]):
            return Y.iloc[0, 0]  # all elements of Y are the same
        if attributes == []:
            return Y.iloc[:, 0].mode()[0]  # return element that occurs the most
        D = self.find_inf_gain_maximizing(attributes, X, Y)  # int
        D_values = pd.unique(X.iloc[:, D])  # possible values in D column
        result = {D: {}}
        attributes.remove(D)
        for D_value in D_values:
            subset = X.loc[X.iloc[:, D] == D_value]
            subset = subset.drop(subset.columns[D], axis=1)  # removing a used column
            result[D][D_value] = self.learn_rec(attributes, X, Y)
        return result


if __name__ == "__main__":

    from ucimlrepo import fetch_ucirepo

    mushroom = fetch_ucirepo(id=73)

    # data
    X_mush = mushroom.data.features
    Y_mush = mushroom.data.targets

    id3 = ID3(X_mush.shape[1], X_mush, Y_mush)
    id3.learn()
    # print(id3.learn())