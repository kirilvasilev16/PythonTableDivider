import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from utilities.divider import Divider


def main():
    train_csv = "input/titanic_encoded_data.csv"
    train_data = pd.read_csv(train_csv, index_col='PassengerId')
    y_column = 'Survived'

    # Initialise divider
    dv = Divider(train_data, y_column)

    # Apply division
    # dv.divide(strategy="correlation", path='output')
    # dv.divide(strategy = "random", path='output')
    # dv.divide(strategy = "random", path='output', onehot=True)
    # dv.divide(strategy='random', path='output', overlap=True)
    dv.divide(strategy='shrink', path='output')

    # Read table names and respective primary keys
    read_tables = dv.read_tables(path='output')
    # Read table contents
    read_tables_contents = dv.read_tables_contents(read_tables=read_tables)
    # Read table connections
    read_tables_connections = dv.read_tables_connections(path='output')

    # Verify correct table division by joining all tables and comparing to original
    # The method can only be used if oneHotEncoding and column overlaps are not used
    print(dv.verify_correctness(train_data=train_data,
                                read_tables_contents=read_tables_contents,
                                read_connections=read_tables_connections))


if __name__ == '__main__':
    for i in range(100):
        print(i)
        main()
