from __future__ import division, print_function

import numpy as np
import pandas as pd

from sacred import Experiment
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
#from sklearn.metrics import f1_score

import fw_data

ex = Experiment('facet_wealth_rfc')

@ex.config
def cfg():
    random_seed = 59
    n_estimators = 100
    criterion = 'gini'
    max_depth = 3
    min_samples_split = 10
    min_samples_leaf = 1
    min_weight_fraction_leaf = 0.
    max_features = 'auto'
    class_weight = 'balanced'
    scoring='roc_auc'
    cv=5

@ex.automain
def run(random_seed,
        n_estimators,
        criterion,
        max_depth,
        min_samples_split,
        min_samples_leaf,
        min_weight_fraction_leaf,
        max_features,
        class_weight,
        scoring,
        cv,
       ):
    X, y, X_unk = fw_data.get_data()
    clf = RandomForestClassifier(criterion=criterion,
                                 n_estimators=n_estimators,
                                 max_depth=max_depth,
                                 min_samples_split=min_samples_split,
                                 min_samples_leaf=min_samples_leaf,
                                 min_weight_fraction_leaf=min_weight_fraction_leaf,
                                 max_features=max_features,
                                 class_weight=class_weight,
                                 random_state=random_seed
                                )
    scores = cross_val_score(clf, X, y, scoring=scoring, cv=cv)
    print('Mean %s score over %d CV-trials is %f +- %f' % (scoring, cv, np.mean(scores), np.std(scores)))

