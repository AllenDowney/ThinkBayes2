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
        data: number of goals scored in a game
        """
        goals = data
        lam = hypo
        like = thinkbayes2.EvalPoissonPmf(goals, lam)
        return like

    def PredictiveDist(self, label='pred'):
        """Computes the distribution of goals scored in a game.

        returns: new Pmf (mixture of Poissons)
        """
        metapmf = thinkbayes2.Pmf()
        for lam, prob in self.Items():
            pred = thinkbayes2.MakePoissonPmf(lam, 15)
            metapmf[pred] = prob

        mix = thinkbayes2.MakeMixture(metapmf, label=label)
        return mix


def main():
    hypos = numpy.linspace(0, 12, 201)

    # start with a prior based on a pseudo observation
    # chosen to yield the right prior mean
    suite1 = Soccer(hypos, label='Germany')
    suite1.Update(0.34)
    suite2 = suite1.Copy(label='Argentina')

    # update with the results of World Cup 2014 final
    suite1.Update(1)
    suite2.Update(0)

    print('posterior mean Germany', suite1.Mean())
    print('posterior mean Argentina', suite2.Mean())

    # plot the posteriors
    thinkplot.PrePlot(2)
    thinkplot.Pdfs([suite1, suite2])
    thinkplot.Show()

    # compute posterior prob Germany is better than Argentina
    post_prob = suite1 > suite2
    print('posterior prob Germany > Argentina', post_prob)

    prior_odds = 1
    post_odds = post_prob / (1 - post_prob)
    k = post_odds / prior_odds
    print('Bayes factor', k)    

    # compute predictive distributions for goals scored in a rematch
    pred1 = suite1.PredictiveDist(label='Germany')
    pred2 = suite2.PredictiveDist(label='Argentina')
    
    # plot the predictive distributions
    thinkplot.PrePlot(2)
    thinkplot.Pdfs([pred1, pred2])
    thinkplot.Show()

    # compute predictive probability of winning rematch
    print('posterior prob Germany wins rematch', pred1 > pred2)
    print('posterior prob Argentina wins rematch', pred2 > pred1)


if __name__ == '__main__':
    main()
