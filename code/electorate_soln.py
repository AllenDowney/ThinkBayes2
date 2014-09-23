"""This file contains code for use with "Think Bayes",
by Allen B. Downey, available from greenteapress.com

Copyright 2014 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
"""

from __future__ import print_function, division

import numpy
import thinkbayes2
import thinkplot


class Electorate(thinkbayes2.Suite):
    """Represents hypotheses about the state of the electorate."""

    def Likelihood(self, data, hypo):
        """Computes the likelihood of the data under the hypothesis.

        hypo: fraction of the population that supports your candidate
        data: poll results
        """
        poll, result = data
        error = result - hypo
        like = poll.ErrorDensity(error)
        return like

    def ProbLose(self):
        total = 0
        for value, prob in self.Items():
            if value < 50:
                total += prob
        return total


class Poll(object):
    """Represents a poll."""

    def __init__(self, bias, std):
        """Construct a poll result."""
        self.bias = bias
        self.std = std

    def ErrorDensity(self, error):
        """Density of the given error in the distribution of error.

        error: difference between actual and poll result
        """
        return thinkbayes2.EvalNormalPdf(error, self.bias, self.std)


def main():
    hypos = numpy.linspace(0, 100, 101)
    suite = Electorate(hypos)

    thinkplot.Pdf(suite, label='prior')

    poll = Poll(1.1, 3.7)
    result = 53
    data = poll, result
    suite.Update(data)

    thinkplot.Pdf(suite, label='posterior')

    print(suite.Mean())
    print(suite.Std())
    print(suite.ProbLose())

    poll = Poll(-2.3, 4.1)
    result = 49
    data = poll, result
    suite.Update(data)

    thinkplot.Pdf(suite, label='posterior2')
    thinkplot.Show()

    print(suite.Mean())
    print(suite.Std())
    print(suite.ProbLose())


if __name__ == '__main__':
    main()
