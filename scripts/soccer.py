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

        hypo: 
        data: 
        """
        like = 1
        return like

    def PredRemaining(self, rem_time, score):
        """Plots the predictive distribution for final number of goals.

        rem_time: remaining time in the game in minutes
        score: number of goals already scored
        """
        # TODO: fill this in


def main():
    hypos = numpy.linspace(0, 12, 201)
    suite = Soccer(hypos)

    thinkplot.Pdf(suite, label='prior')
    print('prior mean', suite.Mean())

    suite.Update(11)
    thinkplot.Pdf(suite, label='posterior 1')
    print('after one goal', suite.Mean())

    thinkplot.Show()


if __name__ == '__main__':
    main()
