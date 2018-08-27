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
        bias, std, result = data
        error = result - hypo
        like = thinkbayes2.EvalNormalPdf(error, bias, std)
        return like


def main():
    hypos = range(0, 101)
    suite = Electorate(hypos)

    thinkplot.PrePlot(3)
    thinkplot.Pdf(suite, label='prior')

    data = 1.1, 3.7, 53
    suite.Update(data)
    thinkplot.Pdf(suite, label='posterior1')
    thinkplot.Save(root='electorate1',
                   xlabel='percentage of electorate',
                   ylabel='PMF',
                   formats=['png'],
                   clf=False)

    print(suite.Mean())
    print(suite.Std())
    print(suite.ProbLess(50))

    data = -2.3, 4.1, 49
    suite.Update(data)

    thinkplot.Pdf(suite, label='posterior2')
    thinkplot.Save(root='electorate2',
                   xlabel='percentage of electorate',
                   ylabel='PMF',
                   formats=['png'])

    print(suite.Mean())
    print(suite.Std())
    print(suite.ProbLess(50))


if __name__ == '__main__':
    main()
