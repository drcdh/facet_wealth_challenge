from __future__ import division, print_function

import pandas as pd


DATA = './data/data.csv'

LABEL_COL = 'Status'
INDEX_COL = 'Id'
INT_COLUMNS = [u'Refreshing',
               u'Runner',
               u'Baker',
               u'Counter',
               u'Regulator',
               u'Has Python',
               u'Has Whiteboard',
               u'Has Reached Balmers Peak',
               u'DNE',
              ]
FLOAT_COLUMNS = [u'Type of Activity Id',
                 u'Type of Movie Id',
                 u'Type of GPU Id',
                 u'Type of Laptop Id',
                 u'Type of Toaster Id',
                 u'Type of Deck Id',
                 u'Type of Whale Id',
                 u'Type of Star Id',
                 u'Type of Dog Id',
                ]
DATA_COLUMNS = INT_COLUMNS + FLOAT_COLUMNS

def get_data(int_transform=True,
             #split_X_y=True,
             #split_train_test=True,
            ):
    data = pd.read_csv(DATA, index_col=INDEX_COL)
    if int_transform:
        data.loc[:, FLOAT_COLUMNS] = (10 * data[FLOAT_COLUMNS]).astype(int)

    is_test = data[LABEL_COL].isnull()
    X_train = data.loc[~is_test, DATA_COLUMNS]
    y_train = data.loc[~is_test, LABEL_COL]
    X_unknown = data.loc[is_test, DATA_COLUMNS]

    return X_train, y_train, X_unknown

