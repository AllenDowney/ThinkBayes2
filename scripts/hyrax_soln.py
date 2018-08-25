"""This file contains code for use with "Think Bayes",
by Allen B. Downey, available from greenteapress.com

Copyright 2014 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
"""

from __future__ import print_function, division

import numpy
import thinkbayes2
import thinkplot


class Hyrax(thinkbayes2.Suite):
    """Represents hypotheses about how many hyraxes there are."""

    def Likelihood(self, data, hypo):
        """Computes the likelihood of the data under the hypothesis.

        hypo: total population
        data: # tagged, # caught (n), # of caught who were tagged (k)
        """
        tagged, n, k = data
        if hypo < tagged + n - k:
            return 0

        p = tagged / hypo
        like = thinkbayes2.EvalBinomialPmf(k, n, p)
        return like


class Hyrax2(thinkbayes2.Suite):
    """Represents hypotheses about how many hyraxes there are."""

    def Likelihood(self, data, hypo):
        """Computes the likelihood of the data under the hypothesis.

        hypo: total population (N)
        data: # tagged (K), # caught (n), # of caught who were tagged (k)
        """
        N = hypo
        K, n, k = data

        if hypo < K + (n - k):
            return 0

        like = thinkbayes2.EvalHypergeomPmf(k, N, K, n)
        return like


def main():
    hypos = range(1, 1000)
    suite = Hyrax(hypos)
    suite2 = Hyrax2(hypos)

    data = 10, 10, 2
    suite.Update(data)
    suite2.Update(data)

    thinkplot.Pdf(suite, label='binomial')
    thinkplot.Pdf(suite, label='hypergeom')
    thinkplot.Show()


if __name__ == '__main__':
    main()
