import pandas as pd


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

