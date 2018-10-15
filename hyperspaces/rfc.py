from numpy import log

from hyperopt import hp

def get_space():
    space = {
        'clf.n_estimators': hp.qloguniform('n_estimators', 2., 4., 1),
        'clf.criterion': hp.choice('criterion', ['gini', 'entropy']),
        'clf.max_depth': hp.quniform('max_depth', 8, 18, 1),
        'clf.min_samples_split': hp.loguniform('min_samples_split', -4, -1),
        'clf.min_samples_leaf': hp.loguniform('min_samples_leaf', -4, -1),
        #'clf.min_weight_fraction_leaf': hp.uniform('min_weight_fraction_leaf', 0., .1),
        'clf.max_features': hp.choice('max_features', ['auto', 'sqrt', 'log2']),
    }
    return space
