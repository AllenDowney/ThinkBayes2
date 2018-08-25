"""This file contains code used in "Think Bayes",
by Allen B. Downey, available from greenteapress.com

Copyright 2014 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
"""

from __future__ import print_function, division

import math
import sys

import thinkbayes2
import thinkplot

FORMATS = ['pdf', 'eps', 'png']


def StrafingSpeed(alpha, beta, x):
    """Computes strafing speed, given location of shooter and impact.

    alpha: x location of shooter
    beta: y location of shooter
    x: location of impact

    Returns: derivative of x with respect to theta
    """
    theta = math.atan2(x - alpha, beta)
    speed = beta / math.cos(theta)**2
    return speed


def MakeLocationPmf(alpha, beta, locations):
    """Computes the Pmf of the locations, given alpha and beta. 

    Given that the shooter is at coordinates (alpha, beta),
    the probability of hitting any spot is inversely proportionate
    to the strafe speed.

    alpha: x position
    beta: y position
    locations: x locations where the pmf is evaluated

    Returns: Pmf object
    """
    pmf = thinkbayes2.Pmf()
    for x in locations:
        prob = 1.0 / StrafingSpeed(alpha, beta, x)
        pmf.Set(x, prob)
    pmf.Normalize()
    return pmf


class Paintball(thinkbayes2.Suite, thinkbayes2.Joint):
    """Represents hypotheses about the location of an opponent."""

    def __init__(self, alphas, betas, locations):
        """Makes a joint suite of parameters alpha and beta.

        Enumerates all pairs of alpha and beta.
        Stores locations for use in Likelihood.

        alphas: possible values for alpha
        betas: possible values for beta
        locations: possible locations along the wall
        """
        self.locations = locations
        pairs = [(alpha, beta) 
                 for alpha in alphas 
                 for beta in betas]
        thinkbayes2.Suite.__init__(self, pairs)

    def Likelihood(self, data, hypo):
        """Computes the likelihood of the data under the hypothesis.

        hypo: pair of alpha, beta
        data: location of a hit

        Returns: float likelihood
        """
        alpha, beta = hypo
        x = data
        pmf = MakeLocationPmf(alpha, beta, self.locations)
        like = pmf.Prob(x)
        return like


def MakePmfPlot(alpha = 10):
    """Plots Pmf of location for a range of betas."""
    locations = range(0, 31)

    betas = [10, 20, 40]
    thinkplot.PrePlot(num=len(betas))

    for beta in betas:
        pmf = MakeLocationPmf(alpha, beta, locations)
        pmf.name = 'beta = %d' % beta
        thinkplot.Pdf(pmf)

    thinkplot.Save('paintball1',
                xlabel='Distance',
                ylabel='Prob',
                formats=FORMATS)


def MakePosteriorPlot(suite):
    """Plots the posterior marginal distributions for alpha and beta.

    suite: posterior joint distribution of location
    """
    marginal_alpha = suite.Marginal(0)
    marginal_alpha.name = 'alpha'
    marginal_beta = suite.Marginal(1)
    marginal_beta.name = 'beta'

    print('alpha CI', marginal_alpha.CredibleInterval(50))
    print('beta CI', marginal_beta.CredibleInterval(50))

    thinkplot.PrePlot(num=2)

    #thinkplot.Pmf(marginal_alpha)
    #thinkplot.Pmf(marginal_beta)
    
    thinkplot.Cdf(thinkbayes2.MakeCdfFromPmf(marginal_alpha))
    thinkplot.Cdf(thinkbayes2.MakeCdfFromPmf(marginal_beta))
    
    thinkplot.Save('paintball2',
                xlabel='Distance',
                ylabel='Prob',
                loc=4,
                formats=FORMATS)


def MakeConditionalPlot(suite):
    """Plots marginal CDFs for alpha conditioned on beta.

    suite: posterior joint distribution of location
    """    
    betas = [10, 20, 40]
    thinkplot.PrePlot(num=len(betas))

    for beta in betas:
        cond = suite.Conditional(0, 1, beta)
        cond.name = 'beta = %d' % beta
        thinkplot.Pdf(cond)

    thinkplot.Save('paintball3',
                xlabel='Distance',
                ylabel='Prob',
                formats=FORMATS)


def MakeContourPlot(suite):
    """Plots the posterior joint distribution as a contour plot.

    suite: posterior joint distribution of location
    """
    thinkplot.Contour(suite.GetDict(), contour=False, pcolor=True)

    thinkplot.Save('paintball4',
                xlabel='alpha',
                ylabel='beta',
                axis=[0, 30, 0, 20],
                formats=FORMATS)


def MakeCrediblePlot(suite):
    """Makes a plot showing several two-dimensional credible intervals.

    suite: Suite
    """
    d = dict((pair, 0) for pair in suite.Values())

    percentages = [75, 50, 25]
    for p in percentages:
        interval = suite.MaxLikeInterval(p)
        for pair in interval:
            d[pair] += 1

    thinkplot.Contour(d, contour=False, pcolor=True)
    thinkplot.Text(17, 4, '25', color='white')
    thinkplot.Text(17, 15, '50', color='white')
    thinkplot.Text(17, 30, '75')

    thinkplot.Save('paintball5',
                   xlabel='alpha',
                   ylabel='beta',
                   formats=FORMATS,
                   legend=False)


def main(script):

    alphas = range(0, 31)
    betas = range(1, 51)
    locations = range(0, 31)

    suite = Paintball(alphas, betas, locations)
    suite.UpdateSet([15, 16, 18, 21])

    MakeCrediblePlot(suite)

    MakeContourPlot(suite)

    MakePosteriorPlot(suite)

    MakeConditionalPlot(suite)

    MakePmfPlot()


if __name__ == '__main__':
    main(*sys.argv)
