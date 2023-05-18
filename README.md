# PythonTableDivider

[![Python 3.8](https://img.shields.io/badge/python-3.8.2-blue.svg)](https://www.python.org/downloads/release/python-380/)


The project contains the code to vertically split a table given the next strategies:
  - Division based on correlation to attribute of interest
  - Randomized division (vanilla random)
  - Randomized division with shrinking of table sizes
  - Randomized division with onehot encoding for leaf table columns
  - Randomized division with column overlapping over leaf tables
  - Randomized division with the possibility to use features are Primary Key - Foreign Key columns

### Prerequisites 
Python 3.8 

### Install 
1. Create virtual environment
`python3 -m venv env`
2. Activate environment
`source env/bin/activate`
3. Install requirements
`pip install -r requirements.txt`

### Run 
Run [main.py](main.py) - It automatically runs all the division strategies with the _titanic_ dataset provided in the [input folder](input).


### Usage
```python
import pandas as pd
from utilities.divider import Divider

# Load the data, assign index column and y column
train_csv = "input/titanic_encoded_data.csv"
train_data = pd.read_csv(train_csv, index_col='PassengerId')
y_column = 'Survived'

# Initialise the divider
dv = Divider(train_data, y_column, path='output')

# Apply division by correlation
dv.divide(strategy="correlation")
# Apply vanilla random division
dv.divide(strategy="random")
# Apply random division with onehot encoding
dv.divide(strategy='random', onehot=4)
# Apply random division with column overlapping
dv.divide(strategy='random', overlap_r=0.5)
# Apply random division with onehot encoding and column overlapping
dv.divide(strategy='random', onehot=4, overlap_r=0.5)
# Apply random division with table shrinking
dv.divide(strategy='shrink')
# Apply random tree division
dv.divide(strategy="random_tree", minimum_columns=4)
# Apply reverse correlation division
dv.divide(strategy="reverse_correlation", minimum_columns=4)
# Apply short reverse correlation division
dv.divide(strategy="short_reverse_correlation", number_children=3, minimum_columns=4)

# Apply all division strategies above simultaneously in 1 LOC
dv.divide_all(overlap_r=0.5, onehot=4)
```
During division, output will be printed in the console, containing the following:
- Table description, where every line is a separate table  with unique index and specified PK column and normal columns. Level of recursion is also shown for every table division and can be used to observe the propagation of the algorithm
- After every division is complete, either True or False will be printed. `True` **must** always be printed if correlation, vanilla random or random with shrinking strategies are used. Otherwise, either True or False are valid outputs.

### Input 
One single csv file. 

### Output
If `divide_all` method is used, in the specified `output` folder, multiple subfolders will be created and each will contain the following files:
 - `tables.json` - Key-value pairs of table name and PK columns. Note that the algorithm considered the `y column` as index column in order to ensure that it will remain part of the base table and not be recursively passed down;
 - `connections.csv` - Connection between tables - every line is in the form `table_name, FK_name, table_name2, PK_name`, which means that `table_name2` has PK column `PK_name` and can be joined with `table_name` table via FK column `FK_name`;
 - Multiple files named `table_X_Y.csv`, where `X` and `Y` are integers. Each file will contain a single table, generated from the table divider. The tables can be used together with `tables.json` and `connections.csv` to apply join operations on them;
 
If `divide` method is used, in the specified `output` folder, no subfolders will be created and the aforementioned files will simply be put there.

### Strategies
  During every vertical split, a primary and a foreign key columns are created, so that it is possible to later join the tables together:
  - Division based on correlation to attribute of interest
    - We pick a column of importance and calculate its correlation to the other columns. The most correlated columns to it are placed in a separate table, then the second most correlated, ... and finally the least correlated. A column of importance is picked for each of the new tables and the algorithm continuous to recursively divide the table vertically.
  - Randomized division (vanilla random)
    - During execution, the input table is randomly split vertically into separate tables and each of the new tables is disjoint to the others (they share no common columns). Next, each of the smaller tables is recursively divided again until we are left with tables, which contain a single PK column and a single column from the initial input table. Those we call leaf tables.
  - Randomized division with shrinking of table sizes 
    - The division is conducted similarly to the vanilla random division with the difference that we check whether it is possible to remove duplicating entries from the recurred tables and therefore shrink their sizes. This allows us to reduce the table size and use the fact that we can rely on PK-FK relation when joining tables.
  - Randomized division with onehot encoding for leaf table columns
    - The division is conducted similarly to the vanilla random division with the difference that the leaf tables can have onehot encoding applied to their single column if: the number of unique values in that column is at most `onehot`.
  - Randomized division with column overlapping over leaf tables
    - The division is conducted similarly to the vanilla random division with the difference that it is possible to have multiple columns appearing in multiple of the smaller tables. When we apply the table division, we randomly sample columns from the initial table and we return `overlap_r` ratio of them to it. That means we sample with replacement. As a result, the same columns can get sampled again and appear in multiple of the child tables.
  - Random Tree Division
    - The division is conducted similarly to the vanilla random division with the exception that the algorithm will try to make use of columns with all unique values to make use of them as PK-FK columns. If none of the columns meet this criteria, then artificial PK-FK columns will be created. Note that there is also a hyperparameter `minimum_columns`, which can set the minimum size of resulting sub-tables. Note that there may still be 1 remaining table with less than `minimum_columns` columns. Furthermore, with the current implementation, the divider will not be verifiable due to the implementation of the verifier. Therefore, you may get `False` as last output, however, the algorithm will still be running the random division correctly.
  - Reverse Correlation Division
    - This division first sorts all the columns based on correlation and tries to distribute them through the subtables in the most uniform way. For example, if we have a list of columns from most correlated to least correlated `[x1, x2, x3, x4, x5, x6, x7, x8]`, then they will be split in the next level as follows: `[x3, x5, x8]`, `[x2, x4, x7]`, and `[x1, x6]`, where `x8`, `x7`, and `x6` will be primary keys respectively. This strategy will continue until the sub-tables have at least `minimum_columns` number of columns. Note that there may still be 1 remaining table with less than `minimum_columns` columns. Furthermore, with the current implementation, the divider will not be verifiable due to the implementation of the verifier. Therefore, you may get `False` as last output, however, the algorithm will still be running the random division correctly.
  - Short Reverse Correlation Division
    - This division strategy is quite different from the aforementioned division strategies. It works as follows:
      1. All features are sorted based on correlation from most correlated to least correlated: `[x1, x2, x3, x4, x5, x6, x7, x8]`;
      2. Hyperparameter `minimum_columns=2` splits the list in chunks `[[x1, x2], [x3, x4], [x5, x6], [x7, x8]]`;
      3. Hyperparameter `number_children=2` constructs the tree between the tables. For instance, table `[x1, x2]` shall become root table with children `[x3, x4]` and `[x5, x6]`, while `[x3, x4]` will become root for `[x7, x8]`;
      4. Primary keys are either assigned from existing columns or synthetic ones are created if no features have unique values;
      5. The assigned primary keys are used to create connections between the subtables;
      6. Note that both subtables `[x3, x4]` and `[x5, x6]` can be joined together since they have the same PK. However, `[x5, x6]` and `[x7, x8]` will not be joinable, because `[x5, x6]` will inherit the assigned primary key of its parent table;
    - The division strategy is considered "short", since it allows subtables to join one another without the need to be directly joined to their parent table;
    - This division strategy will aim to create longer join paths for highly correlated features and thus put them further away from the base table;.
  