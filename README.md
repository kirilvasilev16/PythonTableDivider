# PythonTableDivider

The project contains the code to vertically split a table given the next strategies:
  - Division based on correlation to attribute of interest
  - Randomized division (vanilla random)
  - Randomized division with shrinking of table sizes
  - Randomized division with onehot encoding for leaf table columns
  - Randomized division with column overlapping over leaf tables

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

# Apply all division strategies above simultaneously in 1 LOC
dv.divide_all(overlap_r=0.5, onehot=4)
```
During division, output will be printed in the console, containing the following:
- Table description, where every line is a separate table  with unique index and specified PK column and normal columns. Level of recursion is also shown for every table division and can be used to observe the propagation of the algorithm
- After every division is complete, either True or False will be printed. `True` **must** always be printed if correlation, vanilla random or random with shrinking strategies are used. Otherwise, either True or False are valid outputs.

### Output
If `divide_all` method is used, in the specified `output` folder, multiple subfolders will be created and each will contain the following files:
 - `tables.json` - Key-value pairs of table name and PK columns. Note that the algorithm considered the `y column` as index column in order to ensure that it will remain part of the base table and not be recursively passed down;
 - `connections.csv` - Connection between tables - every line is in the form `table_name, FK_name, table_name2, PK_name`, which means that `table_name2` has PK column `PK_name` and can be joined with `table_name` table via FK column `FK_name`;
 - Multiple files named `table_X_Y.csv`, where `X` and `Y` are integers. Each file will contain a single table, generated from the table divider. The tables can be used together with `tables.json` and `connections.csv` to apply join operations on them;
 
If `divide` method is used, in the specified `output` folder, no subfolders will be created and the aforementioned files will simply be put there.