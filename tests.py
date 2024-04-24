from ucimlrepo import fetch_ucirepo
from ID3 import ID3
import numpy as np
# mushroom = fetch_ucirepo(id=73)

# # data
# X_mush = mushroom.data.features
# Y_mush = mushroom.data.targets


# breast_cancer = fetch_ucirepo(id=14)

# X_bre = breast_cancer.data.features
# Y_bre = breast_cancer.data.targets

def my_test():
    x = np.array([[1,2],[1,4],[2,2], [2,4], [2,3]])
    y = np.array([[1],[1],[0],[0], [1]])
    id3 = ID3(2, x, y)
    id3.learn()
    print(id3.predict(np.array([2,3])))

def test_mushrooms():
    np.random.seed(12345)

    mushroom = fetch_ucirepo(id=73)

    X_mush= mushroom.data.features.to_numpy()
    Y_mush = mushroom.data.targets.to_numpy()
    XY = np.concatenate((X_mush, Y_mush), axis=1).astype(str)
    np.random.shuffle(XY)
    size = Y_mush.size
    X_mush_train = XY[0:int(size*0.6), :-1]
    Y_mush_train = XY[0:int(size*0.6), -1]

    X_mush_predict = XY[int(size*0.6):, :-1]
    Y_mush_predict = XY[int(size*0.6):, -1]


    id3 = ID3(X_mush.shape[1], X_mush_train, Y_mush_train.reshape(Y_mush_train.size, 1))
    id3.learn()
    res = []
    for row in X_mush_predict[:]:
        res.append(id3.predict(row))

    reals = list(y for y in Y_mush_predict)

    size = 0
    difference = []
    for i, j in zip(res, reals):
        if i == j:
            difference.append(True)
            size += 1
        else:
            difference.append(False)
    print(size/len(difference))


def test_breast_cancer():
    np.random.seed(81) # 12,

    breast_cancer = fetch_ucirepo(id=14)

    X_bre = breast_cancer.data.features.to_numpy()
    Y_bre = breast_cancer.data.targets.to_numpy()
    XY = np.concatenate((X_bre, Y_bre), axis=1).astype(str)
    np.random.shuffle(XY)
    size = Y_bre.size
    X_bre_train = XY[0:int(size*0.6), :-1]
    Y_bre_train = XY[0:int(size*0.6), -1]

    X_bre_predict = XY[int(size*0.6):, :-1]
    Y_bre_predict = XY[int(size*0.6):, -1]


    id3 = ID3(X_bre.shape[1], X_bre_train, Y_bre_train.reshape(Y_bre_train.size, 1))
    id3.learn()
    res = []
    for row in X_bre_predict[:]:
        res.append(id3.predict(row))
    reals = list(y for y in Y_bre_predict)
    size = 0
    difference = []
    for i, j in zip(res, reals):
        if i == j:
            difference.append(True)
            size += 1
        else:
            difference.append(False)
    print(size/len(difference))

if __name__ == "__main__":
    # my_test()
    # test_mushrooms()
    test_breast_cancer()
