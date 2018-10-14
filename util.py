from __future__ import division, print_function

import numpy as np

import fw_data

def print_feature_importances(clf, X, y):
    clf.fit(X, y)
    imp = clf.feature_importances_
    imp_sort = np.argsort(imp)
    for i in imp_sort[::-1]:
        print('  %s  -  %f' % (fw_data.DATA_COLUMNS[i], imp[i]))


