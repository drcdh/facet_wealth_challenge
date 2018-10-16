from __future__ import division, print_function

import numpy as np
import pandas as pd
import pickle

from pprint import pformat, pprint

from hyperopt import hp
from hyperopt import fmin, space_eval, tpe, STATUS_OK, Trials
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.utils import shuffle as sk_shuffle

import fw_data
import util

from hyperspaces.rfc import get_space as get_rfc_space

cfg = {
    'random_seed': 59,
    'shuffle': True,
    'onehot': False,
    'class_weight': 'balanced',
    'scoring': 'roc_auc',
    'cv': 10,
    'max_evals': 100,
}

def make_clf(class_weight, random_seed, **kwargs):
    clf = RandomForestClassifier(
            n_estimators=int(kwargs['clf.n_estimators']),
            criterion=kwargs['clf.criterion'],
            max_depth=kwargs['clf.max_depth'],
            min_samples_split=kwargs['clf.min_samples_split'],
            min_samples_leaf=kwargs['clf.min_samples_leaf'],
            #min_weight_fraction_leaf=kwargs['clf.min_weight_fraction_leaf'],
            max_features=kwargs['clf.max_features'],
            class_weight=class_weight,
            random_state=random_seed,
           )
    return clf

def run(
        random_seed,
        shuffle,
        onehot,
        class_weight,
        scoring,
        cv,
        max_evals,
       ):
    X, y, X_unk = fw_data.get_data(onehot=onehot, seed=random_seed)
    if shuffle:
        X, y = sk_shuffle(X, y, random_state=random_seed)
    def objective(hp_args):
        pprint(hp_args)
        clf = make_clf(class_weight, random_seed, **hp_args)
        scores = cross_val_score(clf, X, y, groups=y, scoring=scoring, cv=cv)
        mean_score, std_score = np.mean(scores), np.std(scores)
        print('  Mean  %s  is   %.3f +- %.3f' % (scoring, mean_score, std_score))
        # We want to minimize the negative score
        return {'loss': -mean_score, 'std_loss': std_score, 'status': STATUS_OK, 'clf': clf}
    space = get_rfc_space()
    T = Trials()
    best = fmin(objective, space, algo=tpe.suggest, max_evals=max_evals, trials=T)
    best = space_eval(space, best)
    with open('results/%s' % scoring, 'a') as f:
        f.write('%s = %f\n' % (scoring, -T.losses()[-1]))
        f.write('  %s' % X.shape[1])
        f.write('  %s, onehot=%d, cv=%d, rfc_evals=%d, seed=%d, shuffled=%d\n' % (class_weight, int(onehot), cv, max_evals, random_seed, int(shuffle)))
        pprint(best)
    print('Best model is:')
    pprint(best)
    clf = make_clf(class_weight, random_seed, **best)
    y_pred = cross_val_predict(clf, X, y, groups=y, cv=cv)
    print('Confusion matrix:')
    pprint(confusion_matrix(y, y_pred))
    # Train with entire set
    clf.fit(X, y)
    with open('model.p', 'w') as f:
        pickle.dump(clf, f)
    y_test_pred = clf.predict_proba(X_unk)[:, 1]
    X_unk['Status_Predicted'] = y_test_pred
    X_unk.loc[:, 'Status_Predicted'].to_csv('results/results.csv')
    return best


if __name__ == '__main__':
    run(**cfg)

