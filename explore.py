from __future__ import division, print_function

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fw_data import get_data

def main():
    X, y, X_unk = get_data()

    for c in X.columns:
        al = .5
        Xc = X[c].values
        Xc_unk = X_unk[c].values
        fig, ax = plt.subplots()
        ax.hist(Xc_unk, alpha=al, align='left', label='Test')
        ax.hist(Xc[y==1], alpha=al, align='mid', label='Positive')
        ax.hist(Xc[y==0], alpha=al, align='right', label='Negative')
        ax.set_yscale('log')
        ax.grid()
        ax.legend(loc='best')
        ax.set_xlabel(c)
        fig.savefig(fname='report/images/hist_%s.png' % c, format='png')

if __name__ == '__main__':
    main()
