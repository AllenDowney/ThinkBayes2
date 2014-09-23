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
    """Represents hypotheses about."""

    def Likelihood(self, data, hypo):
        """Computes the likelihood of the data under the hypothesis.

        hypo: 
        data: 
        """
        tagged, n, k = data
        if hypo < tagged + n - k:
            return 0

        p = tagged / hypo
        like = thinkbayes2.EvalBinomialPmf(k, n, p)
        return like


def main():
    hypos = range(1, 1000)
    suite = Hyrax(hypos)

    data = 10, 10, 2
    suite.Update(data)

    thinkplot.Pdf(suite, label='posterior')
    thinkplot.Show()


if __name__ == '__main__':
    main()
