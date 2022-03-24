import pandas as pd
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
    # dv.divide(strategy = "random", overlap_r=0.5)
    # dv.divide(strategy="random", overlap_r=0.5, onehot=4)

    # Apply all division strategies
    dv.divide_all(overlap_r=0.5, onehot=4)


if __name__ == '__main__':
    main()
