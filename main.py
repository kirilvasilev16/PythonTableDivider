import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from utilities.divider import Divider


def main():
    train_csv = "input/titanic_encoded_data.csv"
    train_data = pd.read_csv(train_csv, index_col='PassengerId')
    y_column = 'Survived'

    # train_csv = "input/houses_encoded_data.csv"
    # train_data = pd.read_csv(train_csv, index_col='Id')
    # y_column = 'SalePrice'

    # Initialise divider
    dv = Divider(train_data, y_column, path='output')

    # Apply division
    # dv.divide(strategy="correlation")
    # dv.divide(strategy="random")
    # dv.divide(strategy='shrink')

    # One hot parameter is inclusive
    # dv.divide(strategy = "random", onehot=4)
    # dv.divide(strategy = "random", overlap_r=0.2)

    # Apply all division strategies
    dv.divide(strategy='random', overlap_r=0.2)

    # Verify correctness
    dv.verify()


if __name__ == '__main__':
    main()
    # for i in range(100):
    #     print(i)
    #     main()
