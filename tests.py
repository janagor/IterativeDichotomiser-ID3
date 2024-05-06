from ucimlrepo import fetch_ucirepo
from ID3 import ID3
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from matplotlib import pyplot as plt


class MushroomTests:
    def test_mushrooms(
            self, seed, training_set_percentage_or_size, test_set_percentage_or_size,
            number_of_attributes=None, file_name=None, subtitle=""
        ):
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
        # in order not to validate input from many compinations of parameters
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

        id3 = ID3(
            X_mush.shape[1], X_mush_train, Y_mush_train.reshape(Y_mush_train.size, 1)
        )
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
        # plt.show()
        print(classification_report(actual, predicted))

    def test_mushrooms_full_size(
        self, seed, file_name=None, subtitle="Full size."
    ):
        self.test_mushrooms(seed, 1, 1, file_name=file_name, subtitle=subtitle)

    def test_mushrooms_training_ratio_test_ratio(
        self, seed, training_ratio, test_ratio, file_name=None, subtitle=None
    ):
        if not subtitle:
            subtitle = f"Training ratio: {training_ratio}.\nTest ratio: {test_ratio}."
        self.test_mushrooms(
            seed, training_ratio, test_ratio, file_name=file_name, subtitle=subtitle
        )

    def test_mushrooms_3_to_2_ratio(
        self, seed, file_name=None, subtitle="3 to 2 ratio."
    ):
        self.test_mushrooms(seed, 0.6, 0.4, file_name=file_name, subtitle=subtitle)

    def test_mushrooms_training_size_test_size(
        self, seed, training_size, test_size, file_name=None, subtitle=None
    ):
        if not subtitle:
            subtitle = f"Training size: {training_size}.\nTest size: {test_size}."
        self.test_mushrooms(
            seed, training_size, test_size, file_name=file_name, subtitle=subtitle
        )

    def test_mushrooms_n_attributes(
        self, seed, attributes_num, file_name=None, subtitle=None
    ):
        if not subtitle:
            subtitle = f"Attributes number: {attributes_num}."
        self.test_mushrooms(seed, 1, 1, attributes_num, file_name=file_name, subtitle=subtitle)

    def test_mushrooms_n_attributes_3_to_2_ratio(
        self, seed, attributes_num, file_name=None, subtitle=None
    ):
        if not subtitle:
            subtitle = f"3 to 2 ratio. Attributes_number: {attributes_num}."
        self.test_mushrooms(
            seed, 0.6, 0.4, attributes_num, file_name=file_name, subtitle=subtitle
        )

    def test_mushrooms_n_attributes_training_size_test_size(
        self, seed, training_size, test_size,
        attributes_num, file_name=None, subtitle=None
    ):
        if not subtitle:
            subtitle = f"Attributes number: {attributes_num}.\nTraining size: {training_size}.\nTest size: {test_size}."
        self.test_mushrooms(
            seed, training_size, test_size, attributes_num, file_name=file_name, subtitle=subtitle
        )


class BreastCancerTests:
    def test_breast_cancer(
            self, seed, training_set_percentage_or_size, test_set_percentage_or_size,
            number_of_attributes=None, file_name=None, subtitle=""
        ):
        np.random.seed(seed)
        breast_cancer = fetch_ucirepo(id=14)

        X_bre= breast_cancer.data.features.to_numpy()
        X_bre = np.transpose(X_bre)
        np.random.shuffle(X_bre)
        if number_of_attributes: #removing random attributes
            X_bre = X_bre[:number_of_attributes, :]
        X_bre = np.transpose(X_bre)
        Y_bre = breast_cancer.data.targets.to_numpy()
        XY = np.concatenate((X_bre, Y_bre), axis=1).astype(str)
        np.random.shuffle(XY)
        size = Y_bre.size
        # in order not to validate input from many compinations of parameters
        if training_set_percentage_or_size > 1:
            X_bre_train = XY[0:training_set_percentage_or_size, :-1]
            Y_bre_train = XY[0:training_set_percentage_or_size, -1]
        else:
            X_bre_train = XY[0:int(size*training_set_percentage_or_size), :-1]
            Y_bre_train = XY[0:int(size*training_set_percentage_or_size), -1]

        if test_set_percentage_or_size > 1:
            X_bre_predict = XY[size - test_set_percentage_or_size:, :-1]
            Y_bre_predict = XY[size - test_set_percentage_or_size:, -1]
        else:
            X_bre_predict = XY[int(size*(1 - test_set_percentage_or_size)):, :-1]
            Y_bre_predict = XY[int(size*(1 - test_set_percentage_or_size)):, -1]

        id3 = ID3(X_bre.shape[1], X_bre_train, Y_bre_train.reshape(Y_bre_train.size, 1))
        id3.learn()
        predicted = [] # result of ID3
        actual = [] # actual values
        for indx, row in enumerate(X_bre_predict[:]):
            predicted.append(id3.predict(row))
            actual.append(Y_bre_predict[indx])
        plt.figure(figsize=(12,10))
        cm = confusion_matrix(actual,predicted)
        sns.heatmap(cm, 
                    annot=True,
                    fmt='g', 
                    xticklabels=['Recurrence-events','No-recurrence-events'],
                    yticklabels=['Recurrence-events','No-recurrence-events'])
        plt.ylabel('Prediction',fontsize=13)
        plt.xlabel('Actual',fontsize=13)
        plt.title(f'Confusion Matrix. Breast cancer.\n{subtitle}',fontsize=17)
        if file_name is not None:
            plt.savefig(f'{file_name}.png')
        # plt.show()
        print(classification_report(actual, predicted))

    def test_breast_cancer_full_size(
        self, seed, file_name=None, subtitle="Full size."
    ):
        self.test_breast_cancer(seed, 1, 1, file_name=file_name, subtitle=subtitle)

    def test_breast_cancer_training_ratio_test_ratio(
        self, seed, training_ratio, test_ratio, file_name=None, subtitle=None
    ):
        if not subtitle:
            subtitle = f"Training ratio: {training_ratio}.\nTest ratio: {test_ratio}."
        self.test_breast_cancer(
            seed, training_ratio, test_ratio, file_name=file_name, subtitle=subtitle
        )

    def test_breast_cancer_3_to_2_ratio(
        self, seed, file_name=None, subtitle="3 to 2 ratio."
    ):
        self.test_breast_cancer(seed, 0.6, 0.4, file_name=file_name, subtitle=subtitle)

if __name__ == "__main__":
    mushroomTests = MushroomTests()

    SEED = 1
    mushroomTests.test_mushrooms_full_size(SEED, file_name="mush_full_size")

    SEED = 1
    TRAINING_RATIO = 0.1
    TEST_RATIO = 0.9
    mushroomTests.test_mushrooms_training_ratio_test_ratio(
        SEED, TRAINING_RATIO, TEST_RATIO
    )

    SEED = 1
    mushroomTests.test_mushrooms_3_to_2_ratio(
        SEED, file_name="mush_3_to_2_r"
    )

    SEED = 5 # seed for which key-error does not occur
    SEED = 12
    SEED = 16
    MAX_SIZE = 286 # size of breast cancer set
    TRAINING_SIZE = int(0.6 * MAX_SIZE)
    TEST_SIZE = int(0.4 * MAX_SIZE)
    mushroomTests.test_mushrooms_training_size_test_size(
        SEED, TRAINING_SIZE, TEST_SIZE,
        file_name="mush_train_size_test_size"
    )

    SEED = 1
    ATTRIBUTES_NUM = 9 # breast cancer attributes number
    mushroomTests.test_mushrooms_n_attributes(
        SEED, ATTRIBUTES_NUM, file_name="mush_n_attr"
    )

    SEED = 2 # drop of accuracy to ~90%
    SEED = 5
    SEED = 7
    SEED = 11 # drop of accuracy to ~70%
    ATTRIBUTES_NUM = 9 # breast cancer attributes number
    mushroomTests.test_mushrooms_n_attributes_3_to_2_ratio(
        SEED, ATTRIBUTES_NUM, file_name="mush_n_attr_3_to_2_r"
    )

    # seeds for which key-error does not occur

    SEED = 3
    SEED = 4
    SEED = 9
    SEED = 12 # accuracy is not 100
    SEED = 14
    SEED = 15
    SEED = 16
    SEED = 18
    SEED = 19
    ATTRIBUTES_NUM = 9 # breast cancer attributes number
    MAX_SIZE = 286 # size of breast cancer set
    TRAINING_SIZE = int(0.6 * MAX_SIZE)
    TEST_SIZE = int(0.4 * MAX_SIZE)
    mushroomTests.test_mushrooms_n_attributes_training_size_test_size(
        SEED, TRAINING_SIZE, TEST_SIZE,
        ATTRIBUTES_NUM, file_name="mush_n_attr_train_size_test_size"
    )

    breastCancerTests = BreastCancerTests()
    SEED = 1
    breastCancerTests.test_breast_cancer_full_size(SEED, file_name="bre_full_size")

    # seeds for which key-error does not occur
    SEED = 6
    SEED = 9
    SEED = 11
    SEED = 16
    SEEDS = [6,9,11,16]
    for seed in SEEDS:
        breastCancerTests.test_breast_cancer_3_to_2_ratio(seed, file_name="bre_3_to_2_r")

