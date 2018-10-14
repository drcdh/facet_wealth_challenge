from __future__ import division, print_function

import numpy as np
import pandas as pd

from hyperopt import hp
from hyperopt import fmin, tpe, STATUS_OK
from sacred import Experiment
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
#from sklearn.metrics import f1_score

import fw_data
import util

from hyperspaces.rfc import get_space as get_rfc_space

ex = Experiment('facet_wealth_rfc')

@ex.config
def cfg():
    random_seed = 59
    n_estimators = 1000
    class_weight = 'balanced'
    scoring='roc_auc'
    cv=10
    max_evals=30

@ex.automain
def run(random_seed,
        n_estimators,
        class_weight,
        scoring,
        cv,
        max_evals,
       ):
    X, y, X_unk = fw_data.get_data()
    def objective(args):
        clf = RandomForestClassifier(
                n_estimators=n_estimators,
                criterion=args['clf.criterion'],
                max_depth=args['clf.max_depth'],
                min_samples_split=args['clf.min_samples_split'],
                min_samples_leaf=args['clf.min_samples_leaf'],
                #min_weight_fraction_leaf=args['clf.min_weight_fraction_leaf'],
                max_features=args['clf.max_features'],
                class_weight=class_weight,
                random_state=random_seed,
               )
        scores = cross_val_score(clf, X, y, scoring=scoring, cv=cv)
        losses = [1. - s for s in scores]  # Want to minimize negative score
        return {'loss': np.mean(losses), 'std_loss': np.std(losses), 'status': STATUS_OK}
    space = get_rfc_space()
    best = fmin(objective, space, algo=tpe.suggest, max_evals=max_evals)
    print(best)
#    util.print_feature_importances(clf, X, y)
#    print('Mean %s score over %d CV-trials is %f +- %f' % (scoring, cv, np.mean(scores), np.std(scores)))

