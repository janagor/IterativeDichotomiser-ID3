from __future__ import annotations
from typing import Dict
from collections import Counter
from math import log
from itertools import groupby
import numpy as np
import pandas as pd
from copy import deepcopy
# from sklearn.metrics import confusion_matrix
# import seaborn as sns
# import matplotlib.pyplot as plt

class ID3:
    def __init__(
        self,
        attributes_size: int,
        X: pd.Dataframe,
        Y: pd.Dataframe
    ):
        self.attributes_size = attributes_size
        self.attributes = list(range(attributes_size))
        self.XY = pd.concat([X, Y], axis=1).astype(str)
        self.classes = pd.unique(Y.iloc[:, 0]).astype(str)  # unique values of Y
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
        XY = pd.concat([X, Y], axis=1)
        groups = list(XY.groupby(XY.columns[0]))
        inf_gain = 0
        for _, group in groups:
            inf_gain += (
                group[group.columns[0]].size / size *
                self.countEntropy(group[group.columns])
                )
        return inf_gain

    def find_inf_gain_maximizing(self, attributes, XY) -> int:  # returns the index of column that should by used
        gains = []
        for i in attributes:
            gains.append(self.countInfGain(XY.iloc[:, i], XY.iloc[:, -1]))
        results = list(zip(gains, attributes))
        return max(results, key=lambda result: result[0])[1]

    def learn(self) -> Dict:
        # create copy of parameters
        XY = deepcopy(self.XY)
        attributes = self.attributes
        self.tree = self.learn_rec(attributes, XY)

    def learn_rec(self, attributes, XY) -> Dict:
        if XY.empty:
            raise ValueError  # no elements in X
        temp = XY.iloc[:, -1].to_numpy()
        if (temp[0] == temp).all():
            return temp[0]  # all elements of Y are the same
        if attributes == []:
            return XY.iloc[:, -1].mode()[0]  # return element that occurs the most
        D = self.find_inf_gain_maximizing(attributes, XY)  # int
        D_values = pd.unique(XY.iloc[:, D])  # possible values in D column
        result = {D: {}}
        attributes.remove(D)
        for D_value in D_values:
            subset = XY.loc[XY.iloc[:, D] == D_value]
            result[D][D_value] = self.learn_rec(attributes, subset)
        return result

    def predict(self, sample):
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
            current_node = edges[sample.values[current_index]]
            if type(current_node) is not type({}):
                return current_node




if __name__ == "__main__":

    # from ucimlrepo import fetch_ucirepo

    # mushroom = fetch_ucirepo(id=73)

    # X_mush = mushroom.data.features
    # Y_mush = mushroom.data.targets

    # x = X_mush.iloc[0:-1, :]
    # y = Y_mush.iloc[0:-1, :]


    # id3 = ID3(X_mush.shape[1], X_mush, Y_mush)
    # id3.learn()
    # res = []
    # for _, row in X_mush.iterrows():
    #     res.append(id3.predict(row))

    # reals = list(y[0] for y in Y_mush.to_numpy())

    # size = 0
    # difference = []
    # for i, j in zip(res, reals):
    #     if i == j:
    #         difference.append(True)
    #         size += 1
    #     else:
    #         difference.append(False)
    # print(difference)
    # print(size/len(difference))


    from ucimlrepo import fetch_ucirepo

    breast_cancer = fetch_ucirepo(id=14)

    X_bre = breast_cancer.data.features
    Y_bre = breast_cancer.data.targets

    id3 = ID3(X_bre.shape[1], X_bre, Y_bre)
    id3.learn()
    res = []
    for _, row in X_bre.iterrows():
        res.append(id3.predict(row))
    print(res)
    reals = list(y[0] for y in Y_bre.to_numpy())
    print(id3.tree)
    size = 0
    difference = []
    for i, j in zip(res, reals):
        if i == j:
            difference.append(True)
            size += 1
        else:
            difference.append(False)
    print(difference)
    print(size/len(difference))
