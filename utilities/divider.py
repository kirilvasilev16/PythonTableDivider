import os
import random
import json
import shutil

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from numpy import genfromtxt

# Set warnings to not be displayed
pd.options.mode.chained_assignment = None


class Divider:
    """
    Table divider class
    """

    def __init__(self, input_table, important_column, path):
        """
        Constructor for divider

        input_table - table to apply division on
        important_column - y_column used for predictions
        path - name of output folder to save data
        """
        self.result = []
        self.input_table = input_table.copy()
        self.important_column = important_column
        self.connections = []
        self.index = 0
        self.path = path

    def correlation(self, input_table, important_column, level):
        """
        Recursively cluster most correlated columns to an "important_column"

        input_table - table to apply the clustering on
        important_column - colum of interest, most likely to be Y
        level - level of recursion
        """

        input_table = input_table.copy()

        n_splits = 3

        # Return if no columns
        if len(input_table.columns) == 0:
            return

        # Set up base table variables
        base_index = self.index
        self.index += 1
        base_index_cols = input_table.index.names

        # Calculate correlation between columns and most important column
        corr = abs(input_table.corr(method='spearman'))
        corr = corr.drop([important_column], axis=1)

        # Calculate quantiles based on correlation and n_splits
        quantiles = []
        for i in range(n_splits):
            quantile = 1 - (i + 1) / n_splits
            quantiles.append(corr.loc[[important_column]].T.quantile(quantile)[0])

        # Apply clustering for every cluster of columns
        for threshold in quantiles:
            # Randomly shuffle
            self.index += 1
            mylist = np.array(range(0, len(input_table)))
            random.shuffle(mylist)

            # Break if no columns
            if len(corr.columns) == 0 or len(input_table.columns) == 0:
                break

            # Pick the new important column
            new_important = corr.loc[[important_column]].idxmax(axis=1)[0]
            # Pick all columns with correlation above quantile threshold
            corr_columns = [col for col in corr.loc[[important_column]].columns if
                            corr.loc[[important_column]][col][0] >= threshold]

            if len(corr_columns) == 0:
                continue

            fk_name, mylist, corr_columns, pk_name, recurred_table = self.create_recurred(input_table, level, mylist,
                                                                                          corr_columns)

            # Add new FK and remove columns associated with it
            input_table = self.replace_recurred_with_fk(fk_name, input_table, mylist, corr_columns)
            corr = corr.drop(corr_columns, axis=1)

            # Add the connection to a list
            self.connections.append((f"table_{level}_{base_index}.csv", fk_name,
                                     f"table_{level + 1}_{self.index}.csv", pk_name))

            # Apply clustering recursively
            self.correlation(recurred_table, new_important, level + 1)

        # Reset the index and set FK columns as normal columns
        self.revert_input_table_index(base_index, base_index_cols, input_table, level)

    def revert_input_table_index(self, base_index, base_index_cols, input_table, level):
        """
        Revert the input_table index to previous state

        base_index - identifier of table
        base_index_cols - previous index of input_table
        input_table - table to revert index on
        level - level of recursion
        """
        input_table = input_table.reset_index()
        input_table.set_index(base_index_cols, inplace=True)
        self.result.append((level, base_index, input_table))

    def random_shrink(self, input_table, level):
        """
        Recursively cluster randomly the columns and apply shrinking of the new tables

        input_table - table to apply the clustering on
        level - level of recursion
        """

        input_table = input_table.copy()

        # Return if no columns
        if len(input_table.columns) <= 1:
            return

        # Set up base table variables
        base_index = self.index
        self.index += 1
        base_index_cols = input_table.index.names

        # Apply clustering for every cluster of columns
        while len(input_table.columns) > 0:

            # Initialize recurred table and corresponding fields
            fk_name, mylist, picked_columns, pk_name, recurred_table = self.sample_and_create_recurred(input_table,
                                                                                                       level)

            # Check if table size can be reduced
            unique_recurred_table = recurred_table.drop_duplicates()
            if len(input_table.columns) + len(input_table.index.names) > 2 and len(unique_recurred_table) < len(
                    recurred_table):
                # Add new FK and remove columns associated with it
                old_index = list(input_table.index.names)
                input_table = (
                    input_table.reset_index().merge(unique_recurred_table.reset_index(), on=list(picked_columns.values),
                                                    how='left')
                        .rename(columns={pk_name: fk_name})
                        .groupby(old_index + [fk_name]).first()
                        .drop(picked_columns, axis=1))

                # Set recurred table to have reduced element count
                recurred_table = unique_recurred_table
            else:
                input_table = self.replace_recurred_with_fk(fk_name, input_table, mylist, picked_columns)

            # Add the connection to a list
            self.connections.append((f"table_{level}_{base_index}.csv", fk_name,
                                     f"table_{level + 1}_{self.index}.csv", pk_name))

            if len(recurred_table.columns) == 1:
                self.result.append((level + 1, self.index, recurred_table))
                continue

            # Apply clustering recursively on smaller table
            self.random_shrink(recurred_table, level + 1)

        # Reset the index and set FK columns as normal columns
        self.revert_input_table_index(base_index, base_index_cols, input_table, level)

    def random_same_pk_fk(self, input_table, level, onehot=0, overlap_r=0):
        """
        Recursively cluster randomly the columns

        input_table - table to apply the clustering on
        level - level of recursion
        onehot - parameter for applying onehot encoding
        overlap_r - ratio of columns to overlap
        """

        input_table = input_table.copy()

        # Return if no columns
        if len(input_table.columns) <= 1:
            return

        # Set up base table variables
        base_index = self.index
        self.index += 1
        base_index_cols = input_table.index.names

        # Apply clustering for every cluster of columns
        while len(input_table.columns) > 0:

            # Initialize recurred table and corresponding fields
            fk_name, mylist, picked_columns, pk_name, recurred_table = self.sample_and_create_recurred(input_table,
                                                                                                       level)

            # Add new FK and remove columns associated with it
            input_table = self.replace_recurred_with_fk(fk_name, input_table, mylist, picked_columns)

            # Add the connection to a list
            self.connections.append((f"table_{level}_{base_index}.csv", fk_name,
                                     f"table_{level + 1}_{self.index}.csv", pk_name))

            # Append new FK table to result list
            if len(recurred_table.columns) == 1:
                # Check if you need to apply oneHotEncoding
                if onehot > 0 and len(recurred_table[recurred_table.columns[0]].unique()) <= onehot\
                        and not recurred_table[recurred_table.columns[0]].isnull().values.any():
                    # Apply the onehot encoding
                    recurred_table = self.apply_onehot_encoding(pk_name, mylist, recurred_table)

                # Append single-column table to result
                self.result.append((level + 1, self.index, recurred_table))
                continue

            # Check if you need to apply overlapping and apply it
            elif overlap_r > 0:
                picked_column = recurred_table.sample(n=int(np.floor(len(recurred_table.columns) * overlap_r)),
                                                      axis='columns').columns
                input_table[picked_column] = recurred_table[picked_column].values

            # Apply clustering recursively on smaller table
            self.random_same_pk_fk(recurred_table, level + 1, onehot=onehot, overlap_r=overlap_r)

        # Reset the index and set FK columns as normal columns
        self.revert_input_table_index(base_index, base_index_cols, input_table, level)

    @staticmethod
    def replace_recurred_with_fk(fk_name, input_table, mylist, picked_columns):
        """
        Move FK_column to the index of input table

        fk_name - foreign key column name
        input_table - table to add column to index
        mylist - list of foreign key column values
        picked_columns - drop columns associated with the FK

        returns input table
        """

        input_table.loc[:, fk_name] = mylist
        input_table = input_table.groupby(input_table.index.names + [fk_name]).first()
        input_table = input_table.drop(picked_columns, axis=1)
        return input_table

    def sample_and_create_recurred(self, input_table, level):
        """
        Initialize recurred table by sampling columns from input table
        Initialize PK and FK fields as well

        input_table - table to sample from
        level - level of recursion

        returns foreign key name, PK column values, sampled columns, PK column name, the recurred table
        """

        # Randomly shuffle
        self.index += 1
        mylist = np.array(range(0, len(input_table)))
        random.shuffle(mylist)

        # Randomly pick n_splits columns
        picked_columns = []
        if len(input_table.columns) > 1:
            picked_columns = input_table.sample(n=np.random.randint(1, len(input_table.columns)),
                                                axis='columns').columns
        else:
            picked_columns = input_table.columns

        return self.create_recurred(input_table, level, mylist, picked_columns)

    def create_recurred(self, input_table, level, mylist, picked_columns):
        """
        Create the recurred table

        input_table - table to sample from
        level - level of recursion
        mylist - PK values
        picked_columns - columns sampled for the recurred table

        returns foreign key name, PK column values, sampled columns, PK column name, the recurred table
        """

        # Set new column names
        pk_name = 'PK_' + str(level + 1) + '_' + str(self.index)
        fk_name = 'FK_' + str(level + 1) + '_' + str(self.index)
        # Add new PK
        recurred_table = input_table[picked_columns]
        recurred_table.loc[:, pk_name] = mylist
        recurred_table.set_index(pk_name, inplace=True)
        return fk_name, mylist, picked_columns, pk_name, recurred_table

    @staticmethod
    def apply_onehot_encoding(pk_name, mylist, recurred_table):
        """
        Applies onehot encoding on the first column of a table

        pk_name - Primary key column name
        mylist - list of values used as PK in the recurred table index
        recurred_table - the table to apply the encoding on

        returns recurred table with onehot encoding applied on it
        """
        oneHotEncoder = OneHotEncoder()
        encoded_col = pd.DataFrame(
            oneHotEncoder.fit_transform(recurred_table[[recurred_table.columns[0]]]).toarray())
        # Rename columns to have prefix
        oneHotEncoded_column_name = recurred_table.columns[0]
        encoded_col = encoded_col.add_prefix(oneHotEncoded_column_name)
        # Concatenate the 2 tables
        recurred_table = pd.concat([recurred_table.reset_index(), encoded_col], axis=1, copy=False,
                                   join='inner')
        recurred_table.drop([oneHotEncoded_column_name], axis=1, inplace=True)
        # Readd the index column
        recurred_table.loc[:, pk_name] = mylist
        recurred_table.set_index(pk_name, inplace=True)
        return recurred_table

    def divide(self, strategy, onehot=False, overlap_r=0):
        """
        Function used to divide the table

        strategy - strategy of division
        onehot - parameter for applying onehot encoding
        overlap_r - ratio of columns to overlap
        """

        # Delete the output folder and make a new one
        shutil.rmtree(self.path, ignore_errors=True)
        os.makedirs(self.path, exist_ok=True)

        # Initialise fresh result and connections lists
        self.__init__(self.input_table, self.important_column, self.path)

        # Pick strategy
        if strategy == 'random':
            input_table = self.input_table.groupby(self.input_table.index.names + [self.important_column]).first()
            self.random_same_pk_fk(input_table, 0, onehot=onehot, overlap_r=overlap_r)
        elif strategy == 'shrink':
            input_table = self.input_table.groupby(self.input_table.index.names + [self.important_column]).first()
            self.random_shrink(input_table, 0)
        elif strategy == 'correlation':
            self.correlation(self.input_table, self.important_column, 0)

        # Sort result by recursion level and index
        self.result.sort(key=lambda x: (x[0], x[1]))

        # Print results and save every table to a file
        print('Level, Index, Primary Key, Columns')
        for (el, col, table) in self.result:
            print(el, " ", col, " ", table.index.names, " ", table.columns, "\n")
            table.to_csv(self.path + '/table_' + str(el) + '_' + str(col) + '.csv')

        # Initialise set of tuples in the form (table name, PK column)
        all_tables = []

        # Iterate over tables and fill data
        for (el, col, table) in self.result:
            # Add tables to set
            all_tables.append((f"table_{el}_{col}.csv", table.index.names))

        # Save connections to file
        np.savetxt(self.path + "/connections.csv", self.connections, delimiter=',', fmt='%s')

        # Save tables names with their PK to file
        all_tables = json.dumps(all_tables)
        with open(self.path + '/tables.json', 'w') as outfile:
            json.dump(all_tables, outfile)

        # Verify correctness
        self.verify()

    def read_tables(self):
        """
        Read data from tables

        returns array of [table name, Primary key]
        """
        with open(self.path + '/tables.json') as json_file:
            tables = json.loads(json.load(json_file))
        return tables

    def read_tables_contents(self, read_tables):
        """
        Read the actual data from the tables

        returns the contents of the tab;es
        """
        read_tables_content = dict()
        for [table, pk] in read_tables:
            read_tables_content[table] = pd.read_csv(self.path + '/' + table, index_col=pk)
        return read_tables_content

    def read_tables_connections(self):
        """
        Read table connections

        returns list with members in the form ((table_name1, PK1), (table_name2, FK1))
        """
        read_connections = map(lambda x: ((x[0], x[1]), (x[2], x[3])),
                               genfromtxt(self.path + '/connections.csv', delimiter=',', dtype='str'))
        return list(read_connections)

    @staticmethod
    def verify_correctness(train_data, read_tables_contents, read_connections):
        """
        Join all tables and compare to original table to verify that the result will be the same in the end

        The method can only be used if oneHotEncoding and column overlaps are not used

        train_data - input table
        read_tables_contents - divided tables read from the output files
        read_connections - PK-FK connections between tables

        Method must always return True for vanilla random, correlation and random with shrinking strategies
        Method may return True or False for rest
        Method should be called after every division strategy
        """
        ((table1, index1), (table2, index2)) = read_connections[0]
        joined_table = read_tables_contents[table1]
        old_index = list(joined_table.index.names)

        joined_table = (joined_table.reset_index().merge(read_tables_contents[table2], left_on=index1, right_on=index2)
                        .groupby(old_index).first()
                        .drop([index1], axis=1))

        for i in range(len(read_connections) - 1):
            ((table1, index1), (table2, index2)) = read_connections[i + 1]

            old_index = list(joined_table.index.names)

            joined_table = (
                joined_table.reset_index().merge(read_tables_contents[table2], left_on=index1, right_on=index2)
                    .groupby(old_index).first()
                    .drop([index1], axis=1))

        return joined_table.reset_index().sort_index().sort_index(axis=1).equals(
            train_data.reset_index().sort_index().sort_index(axis=1))

    def verify(self):
        """
        Calls all the verification methods to verify correctness
        In case all leaf tables are joined together in one table, and it is identical to the input table, returns True
        Otherwise returns False as joined table contain different data
        """

        # Read table names and respective primary keys
        read_tables = self.read_tables()
        # Read table contents
        read_tables_contents = self.read_tables_contents(read_tables=read_tables)
        # Read table connections
        read_tables_connections = self.read_tables_connections()

        # Verify correct table division by joining all tables and comparing to original
        # The method can only be used if oneHotEncoding and column overlaps are not used
        print(self.verify_correctness(train_data=self.input_table,
                                      read_tables_contents=read_tables_contents,
                                      read_connections=read_tables_connections))

    def divide_all(self, overlap_r, onehot):
        """
        Apply all division strategies

        overlap_p - probability of randomly overlapping columns
        overlap_r - ratio of columns to overlap
        """
        path = self.path
        print("Random")
        self.path = path + "/random"
        self.divide("random")

        print("Random with onehot")
        self.path = path + "/random_onehot"
        self.divide("random", onehot=onehot)

        print("Random with overlap")
        self.path = path + "/random_overlap"
        self.divide("random", overlap_r=overlap_r)

        print("Random with onehot and overlap")
        self.path = path + "/random_onehot_overlap"
        self.divide("random", onehot=onehot, overlap_r=overlap_r)

        print("Random with shrinking")
        self.path = path + "/shrink"
        self.divide("shrink")

        print("Correlation based")
        self.path = path + "/correlation"
        self.divide("correlation")

        self.path = path
