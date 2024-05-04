from ucimlrepo import fetch_ucirepo
from ID3 import ID3
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from matplotlib import pyplot as plt


def test_mushrooms(
        seed, training_set_percentage_or_size, test_set_percentage_or_size,
        number_of_attributes=None, file_name=None, subtitle=""
    ):
    print(subtitle)
    np.random.seed(seed)
    mushroom = fetch_ucirepo(id=73)

    X_mush= mushroom.data.features.to_numpy()
    X_mush = np.transpose(X_mush)
    np.random.shuffle(X_mush)
    if number_of_attributes: #removing random attributes
        X_mush = X_mush[:number_of_attributes, :]
        X_mush = np.transpose(X_mush)
    Y_mush = mushroom.data.targets.to_numpy()
    XY = np.concatenate((X_mush, Y_mush), axis=1).astype(str)
    np.random.shuffle(XY)
    size = Y_mush.size
    if training_set_percentage_or_size > 1:
        X_mush_train = XY[0:training_set_percentage_or_size, :-1]
        Y_mush_train = XY[0:training_set_percentage_or_size, -1]
    else:
        X_mush_train = XY[0:int(size*training_set_percentage_or_size), :-1]
        Y_mush_train = XY[0:int(size*training_set_percentage_or_size), -1]

    if test_set_percentage_or_size > 1:
        X_mush_predict = XY[size - test_set_percentage_or_size:, :-1]
        Y_mush_predict = XY[size - test_set_percentage_or_size:, -1]
    else:
        X_mush_predict = XY[int(size*(1 - test_set_percentage_or_size)):, :-1]
        Y_mush_predict = XY[int(size*(1 - test_set_percentage_or_size)):, -1]

    id3 = ID3(X_mush.shape[1], X_mush_train, Y_mush_train.reshape(Y_mush_train.size, 1))
    id3.learn()
    predicted = [] # result of ID3
    actual = [] # actual values
    for indx, row in enumerate(X_mush_predict[:]):
        predicted.append(id3.predict(row))
        actual.append(Y_mush_predict[indx])
    plt.figure(figsize=(12,10))
    cm = confusion_matrix(actual,predicted)
    sns.heatmap(cm, 
                annot=True,
                fmt='g', 
                xticklabels=['Poisonous','Not poisonous'],
                yticklabels=['Poisonous','Not poisonous'])
    plt.ylabel('Prediction',fontsize=13)
    plt.xlabel('Actual',fontsize=13)
    plt.title(f'Confusion Matrix. Mushrooms.\n{subtitle}',fontsize=17)
    if file_name is not None:
        plt.savefig(f'{file_name}.png')
    plt.show()
    print(classification_report(actual, predicted))


def test_mushrooms_full_size(seed, subtitle="Full size."):
    test_mushrooms(seed, 1, 1, subtitle=subtitle)


def test_mushrooms_training_ratio_test_ratio(seed, training_ratio, test_ratio, file_name=None, subtitle=None):
    if not subtitle:
        subtitle = f"Training ratio: {training_ratio}.\nTest ratio: {test_ratio}."
    test_mushrooms(seed, training_ratio, test_ratio, file_name=file_name, subtitle=subtitle)


def test_mushrooms_3_to_2_ratio(seed, file_name=None, subtitle="3 to 2 ratio."):
    test_mushrooms(seed, 0.6, 0.4, file_name=file_name, subtitle=subtitle)


def test_mushrooms_training_size_test_size(seed, training_size, test_size, file_name=None, subtitle=None):
    if not subtitle:
        subtitle = f"Training size: {training_size}.\nTest size: {test_size}."
    test_mushrooms(seed, training_size, test_size, file_name=file_name, subtitle=subtitle)


def test_mushrooms_n_attributes(seed, attributes_num, file_name=None, subtitle=None):
    if not subtitle:
        subtitle = f"Attributes number: {attributes_num}."
    test_mushrooms(seed, 1, 1, attributes_num, file_name=file_name, subtitle=subtitle)


def test_mushrooms_n_attributes_3_to_2_ratio(seed, attributes_num, file_name=None, subtitle=None):
    if not subtitle:
        subtitle = f"3 to 2 ratio. Attributes_number: {attributes_num}."
    test_mushrooms(seed, 0.6, 0.4, attributes_num, file_name=file_name, subtitle=subtitle)


def test_mushrooms_n_attributes_training_size_test_size(seed, training_size, test_size, attributes_num, file_name=None, subtitle=None):
    if not subtitle:
        subtitle = f"Attributes number: {attributes_num}.\nTraining size: {training_size}.\nTest size: {test_size}."
    test_mushrooms(seed, training_size, test_size, attributes_num, file_name=file_name, subtitle=subtitle)
###############################################################################
def test_breast_cancer(
        SEED, TRAINING_SET_PERCENTAGE, TEST_SET_PERCENTAGE,
        message="Breast cancer results: ", picture_name = None
    ):
    np.random.seed(SEED)

    breast_cancer = fetch_ucirepo(id=14)

    X_bre = breast_cancer.data.features.to_numpy()
    Y_bre = breast_cancer.data.targets.to_numpy()
    XY = np.concatenate((X_bre, Y_bre), axis=1).astype(str)
    np.random.shuffle(XY)
    size = Y_bre.size
    X_bre_train = XY[0:int(size*TRAINING_SET_PERCENTAGE), :-1]
    Y_bre_train = XY[0:int(size*TRAINING_SET_PERCENTAGE), -1]

    X_bre_predict = XY[int(size*(1 - TEST_SET_PERCENTAGE)):, :-1]
    Y_bre_predict = XY[int(size*(1 - TEST_SET_PERCENTAGE)):, -1]


    id3 = ID3(X_bre.shape[1], X_bre_train, Y_bre_train.reshape(Y_bre_train.size, 1))
    id3.learn()
    predicted = [] # result of ID3
    actual = [] # actual values
    for indx, row in enumerate(X_bre_predict[:]):
        predicted.append(id3.predict(row))
        actual.append(Y_bre_predict[indx])

    cm = confusion_matrix(actual,predicted)
    sns.heatmap(cm, 
                annot=True,
                fmt='g', 
                xticklabels=['Recurrence-events','No-recurrence-events'],
                yticklabels=['Recurrence-events','No-recurrence-events'])
    plt.ylabel('Prediction',fontsize=13)
    plt.xlabel('Actual',fontsize=13)
    plt.title('Confusion Matrix',fontsize=17)
    plt.show()
    print(classification_report(actual, predicted))

def test_breast_cancer_full_size(SEED, message="Breast cancer full size results: "):
    test_breast_cancer(SEED, 1, 1, message)

def test_breast_cancer_3_to_2_ratio(SEED, message="Breast cancer 3 to 2 ration results: "):
    test_breast_cancer(SEED, 0.6, 0.4, message)


###############################################################################


def test_breast_cancers():
    np.random.seed(81) # 12,

    breast_cancer = fetch_ucirepo(id=14)

    X_bre = breast_cancer.data.features.to_numpy()
    Y_bre = breast_cancer.data.targets.to_numpy()
    XY = np.concatenate((X_bre, Y_bre), axis=1).astype(str)
    np.random.shuffle(XY)
    size = Y_bre.size
    X_bre_train = XY[0:int(size), :-1]
    Y_bre_train = XY[0:int(size), -1]

    X_bre_predict = XY[0:int(size), :-1]
    Y_bre_predict = XY[0:int(size), -1]


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
    #my_test()
    #test_mushroomss()
    #test_breast_cancer()
    test_mushrooms_n_attributes_training_size_test_size(123456, 4000, 2000, 15,file_name="temp")

    #test_mushrooms(123456, 1, 1, 10)
    #test_breast_cancer(12345, 1, 1, message="Breast cancer results: ")
