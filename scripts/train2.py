"""This file contains code for use with "Think Bayes",
by Allen B. Downey, available from greenteapress.com

Copyright 2012 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
"""

from __future__ import print_function

from dice import Dice
import thinkplot

class Train(Dice):
    """The likelihood function for the train problem is the same as
    for the Dice problem."""


def Mean(suite):
    total = 0
    for hypo, prob in suite.Items():
        total += hypo * prob
    return total


def MakePosterior(high, dataset):
    hypos = range(1, high+1)
    suite = Train(hypos)
    suite.name = str(high)

    for data in dataset:
        suite.Update(data)

    thinkplot.Pmf(suite)
    return suite


def main():
    dataset = [30, 60, 90]

    for high in [500, 1000, 2000]:
        suite = MakePosterior(high, dataset)
        print(high, suite.Mean())

    thinkplot.Save(root='train2',
                   xlabel='Number of trains',
                   ylabel='Probability')


if __name__ == '__main__':
    main()
