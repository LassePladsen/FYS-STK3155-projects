import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split


def get_exoplanet_data(
    trainfile: str = "../data/exoTrain.csv", testfile: str = "../data/exoTest.csv"
) -> (pd.DataFrame, pd.DataFrame, pd.Series, pd.Series):
    """Returns the exoplanet training and testing data from the data folder"""

    data_train = pd.read_csv(trainfile)
    data_test = pd.read_csv(testfile)

    x_train = data_train.drop("LABEL", axis=1)
    y_train = data_train["LABEL"] - 1  # Replace all ones with zeros and twos with ones

    x_test = data_test.drop("LABEL", axis=1)
    y_test = data_test["LABEL"] - 1  # Replace all ones with zeros and twos with ones

    return x_train, x_test, y_train, y_test


def get_balanced_exoplanet_data(
    trainfile: str = "../data/exoTrain.csv", testfile: str = "../data/exoTest.csv"
) -> (pd.DataFrame, pd.DataFrame, pd.Series, pd.Series):
    """Returns the exoplanet training and testing data from the data folder oversampled using the SMOTE algorithm from imblearn, then split into new train and test split using scikit-learns' train_test_split"""
    x_train, x_test, y_train, y_test = get_exoplanet_data(trainfile, testfile)

    # SMOTE AND SPLIT
    smote = SMOTE()
    x_train_smote, y_train_smote = smote.fit_resample(x_train, y_train)
    return train_test_split(x_train_smote, y_train_smote, test_size=0.2)
