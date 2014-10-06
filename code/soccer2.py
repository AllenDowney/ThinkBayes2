"""This file contains code for use with "Think Bayes",
by Allen B. Downey, available from greenteapress.com

Copyright 2014 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
"""

from __future__ import print_function, division

import numpy
import thinkbayes2
import thinkplot


class Soccer(thinkbayes2.Suite):
    """Represents hypotheses about."""

    def Likelihood(self, data, hypo):
        """Computes the likelihood of the data under the hypothesis.

        hypo: goal rate in goals per game
        data: interarrival time in minutes
        """
        goals = data
        lam = hypo
        like = thinkbayes2.EvalPoissonPmf(goals, lam)
        return like


def main():
    hypos = numpy.linspace(0, 12, 201)
    suite = Soccer(hypos)

    # start with a prior based on the mean interarrival time
    suite.Update(0.34)
    thinkplot.Pdf(suite, label='prior')
    print('prior mean', suite.Mean())
    thinkplot.Show()

    return
    suite.Update(11)
    thinkplot.Pdf(suite, label='posterior 1')
    print('after one goal', suite.Mean())

    suite.Update(12)
    thinkplot.Pdf(suite, label='posterior 2')
    print('after two goals', suite.Mean())

    thinkplot.Show()

    # plot the predictive distribution
    suite.PredRemaining(90-23, 2)


if __name__ == '__main__':
    main()
