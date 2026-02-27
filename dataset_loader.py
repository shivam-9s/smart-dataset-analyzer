import pandas as pd
from sklearn.datasets import (
    load_iris,
    load_wine,
    load_breast_cancer,
    load_diabetes,
    fetch_california_housing
)

def load_dataset(name):
    if name == "Iris":
        data = load_iris()
    elif name == "Wine":
        data = load_wine()
    elif name == "Breast Cancer":
        data = load_breast_cancer()
    elif name == "Diabetes":
        data = load_diabetes()
    elif name == "California Housing":
        data = fetch_california_housing()
    else:
        return None, None

    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)

    return X, y
