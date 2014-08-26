"""This file contains code for use with "Think Bayes",
by Allen B. Downey, available from greenteapress.com

Copyright 2012 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
"""

from __future__ import print_function, division

import random

import thinkbayes2
import thinkplot

FORMATS = ['pdf', 'eps', 'png']


class Die(thinkbayes2.Pmf):
    """Represents the PMF of outcomes for a die."""

    def __init__(self, sides, label=''):
        """Initializes the die.

        sides: int number of sides
        label: string
        """
        hypos = range(1, sides+1)
        thinkbayes2.Pmf.__init__(self, hypos, label=label)


def PmfMax(pmf1, pmf2):
    """Computes the distribution of the max of values drawn from two Pmfs.

    pmf1, pmf2: Pmf objects

    returns: new Pmf
    """
    res = thinkbayes2.Pmf()
    for v1, p1 in pmf1.Items():
        for v2, p2 in pmf2.Items():
            res.Incr(max(v1, v2), p1*p2)
    return res
    

def main():
    pmf_dice = thinkbayes2.Pmf()
    pmf_dice.Set(Die(4), 5)
    pmf_dice.Set(Die(6), 4)
    pmf_dice.Set(Die(8), 3)
    pmf_dice.Set(Die(12), 2)
    pmf_dice.Set(Die(20), 1)
    pmf_dice.Normalize()

    mix = thinkbayes2.Pmf()
    for die, weight in pmf_dice.Items():
        for outcome, prob in die.Items():
            mix.Incr(outcome, weight*prob)

    mix = thinkbayes2.MakeMixture(pmf_dice)

    thinkplot.Hist(mix, width=0.9)
    thinkplot.Save(root='dungeons3',
                xlabel='Outcome',
                ylabel='Probability',
                formats=FORMATS)

    random.seed(17)

    d6 = Die(6, 'd6')

    dice = [d6] * 3
    three = thinkbayes2.SampleSum(dice, 1000)
    three.label = 'sample'
    three.Print()

    three_exact = d6 + d6 + d6
    three_exact.label = 'exact'
    three_exact.Print()

    thinkplot.PrePlot(num=2)
    thinkplot.Pmf(three)
    thinkplot.Pmf(three_exact, linestyle='dashed')
    thinkplot.Save(root='dungeons1',
                xlabel='Sum of three d6',
                ylabel='Probability',
                axis=[2, 19, 0, 0.15],
                formats=FORMATS)

    thinkplot.Clf()
    thinkplot.PrePlot(num=1)
    
    # compute the distribution of the best attribute the hard way
    best_attr2 = PmfMax(three_exact, three_exact)
    best_attr4 = PmfMax(best_attr2, best_attr2)
    best_attr6 = PmfMax(best_attr4, best_attr2)
    # thinkplot.Pmf(best_attr6)

    # and the easy way
    best_attr_cdf = three_exact.Max(6)
    best_attr_cdf.label = ''
    best_attr_pmf = best_attr_cdf.MakePmf()
    best_attr_pmf.Print()

    thinkplot.Pmf(best_attr_pmf)
    thinkplot.Save(root='dungeons2',
                   xlabel='Sum of three d6',
                   ylabel='Probability',
                   axis=[2, 19, 0, 0.23],
                   formats=FORMATS,
                   legend=False)
    


if __name__ == '__main__':
    main()
