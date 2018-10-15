from __future__ import division, print_function

import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder

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
             onehot=False,
             seed=None,
            ):
    data = pd.read_csv(DATA, index_col=INDEX_COL)
    X_int = data.loc[:, INT_COLUMNS]
    X_float = data.loc[:, FLOAT_COLUMNS]
    y = data.loc[:, LABEL_COL]
    if int_transform:
        X_float.loc[:, FLOAT_COLUMNS] = (10 * X_float.loc[:, FLOAT_COLUMNS]).astype(int)
    if onehot:
        enc = OneHotEncoder(dtype=np.int)
        enc_d = enc.fit_transform(X_float.values)
        X_float = pd.DataFrame(
                  enc_d.toarray(),
                  index=X_float.index,
                 )
    is_test = y.isnull()
    X_train = pd.concat([X_int.loc[~is_test],
                         X_float.loc[~is_test]], axis=1)
    y_train = y.loc[~is_test]
    X_unknown = pd.concat([X_int.loc[is_test],
                           X_float.loc[is_test]], axis=1)

    return X_train, y_train, X_unknown

