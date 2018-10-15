from __future__ import division, print_function

import numpy as np
import pandas as pd

from pprint import pformat, pprint

from hyperopt import hp
from hyperopt import fmin, space_eval, tpe, STATUS_OK, Trials
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle as sk_shuffle

import fw_data
import util

from hyperspaces.rfc import get_space as get_rfc_space

cfg = {
    'random_seed': 59,
    'shuffle': True,
    'onehot': False,
    'class_weight': 'balanced',
    'imp_cut': .0,
    'scoring': 'roc_auc',
    'scoring_buffer': .05,
    'cv': 10,
    'max_evals': 1000,
    'n_rec': 1,
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

def subtrial_run(
        random_seed,
        shuffle,
        onehot,
        class_weight,
        imp_cut,
        scoring,
        cv,
        max_evals,
        feature_importances=None,
       ):
    X, y, X_unk = fw_data.get_data(onehot=onehot, seed=random_seed)
    if feature_importances is not None and imp_cut > 0.:
        print('Cutting by feature importance %f' % imp_cut)
        feature_mask = feature_importances <= imp_cut
        X = X.loc[:, feature_mask]
        X_unk = X_unk.loc[:, feature_mask]
        print('  Number of features is now %d' % X.shape[1])
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
        f.write('  %s, onehot=%d, imp_cut=%f, cv=%d, rfc_evals=%d, seed=%d, shuffled=%d\n' % (class_weight, int(onehot), imp_cut, cv, max_evals, random_seed, int(shuffle)))
        pprint(best)
    return best


def run(
        random_seed,
        shuffle,
        onehot,
        class_weight,
        imp_cut,
        scoring,
        scoring_buffer,
        cv,
        max_evals,
        n_rec,
       ):
    print('Should do %d subruns' % n_rec)
    X, y, X_unk = fw_data.get_data(onehot, seed=random_seed)
    score, last_score = None, None
    best = {}
    clf = None
    feature_importances = None
    while n_rec > 0:
        print('Recursive (meta-feature analysis) step: %d' % n_rec)
        if (last_score is None or score > last_score - scoring_buffer) and n_rec > 0:
            last_score = score
            best = subtrial_run(
                random_seed,
                shuffle,
                onehot,
                class_weight,
                imp_cut,
                scoring,
                cv,
                max_evals,
                feature_importances=feature_importances,
            )
            clf = make_clf(class_weight, random_seed, **best).fit(X, y)
            score = cross_val_score(clf, X, y, scoring=scoring)
            score = np.mean(score)
            print('Got  %s = %.4f after %.4f' % (scoring, score, (last_score or -1.)))
            feature_importances = clf.feature_importances_
        else:
            print('Stop')
            break
        n_rec -= 1

if __name__ == '__main__':
    run(**cfg)

