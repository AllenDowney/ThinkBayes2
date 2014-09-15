"""This file contains code used in "Think Bayes",
by Allen B. Downey, available from greenteapress.com

Copyright 2012 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
"""

from __future__ import print_function, division

import math
import numpy
import pickle
import numpy
import random
import scipy

import brfss

import thinkplot
import thinkbayes2

import matplotlib.pyplot as pyplot


NUM_SIGMAS = 1

class Height(thinkbayes2.Suite, thinkbayes2.Joint):
    """Hypotheses about parameters of the distribution of height."""

    def __init__(self, mus, sigmas, label=None):
        """Makes a prior distribution for mu and sigma based on a sample.

        mus: sequence of possible mus
        sigmas: sequence of possible sigmas
        label: string label for the Suite
        """
        pairs = [(mu, sigma) 
                 for mu in mus
                 for sigma in sigmas]

        thinkbayes2.Suite.__init__(self, pairs, label=label)

    def Likelihood(self, data, hypo):
        """Computes the likelihood of the data under the hypothesis.

        Args:
            hypo: tuple of hypothetical mu and sigma
            data: float sample

        Returns:
            likelihood of the sample given mu and sigma
        """
        x = data
        mu, sigma = hypo
        like = scipy.stats.norm.pdf(x, mu, sigma)
        return like

    def LogLikelihood(self, data, hypo):
        """Computes the log likelihood of the data under the hypothesis.

        Args:
            data: a list of values
            hypo: tuple of hypothetical mu and sigma

        Returns:
            log likelihood of the sample given mu and sigma (unnormalized)
        """
        x = data
        mu, sigma = hypo
        loglike = EvalNormalLogPdf(x, mu, sigma)
        return loglike

    def LogUpdateSetFast(self, data):
        """Updates the suite using a faster implementation.

        Computes the sum of the log likelihoods directly.

        Args:
            data: sequence of values
        """
        xs = tuple(data)
        n = len(xs)

        for hypo in self.Values():
            mu, sigma = hypo
            total = Summation(xs, mu)
            loglike = -n * math.log(sigma) - total / 2 / sigma**2
            self.Incr(hypo, loglike)

    def LogUpdateSetMeanVar(self, data):
        """Updates the suite using ABC and mean/var.

        Args:
            data: sequence of values
        """
        xs = data
        n = len(xs)

        m = numpy.mean(xs)
        s = numpy.std(xs)

        self.LogUpdateSetABC(n, m, s)

    def LogUpdateSetMedianIPR(self, data):
        """Updates the suite using ABC and median/iqr.

        Args:
            data: sequence of values
        """
        xs = data
        n = len(xs)

        # compute summary stats
        median, s = MedianS(xs, num_sigmas=NUM_SIGMAS)
        print('median, s', median, s)

        self.LogUpdateSetABC(n, median, s)

    def LogUpdateSetABC(self, n, m, s):
        """Updates the suite using ABC.

        n: sample size
        m: estimated central tendency
        s: estimated spread
        """
        for hypo in sorted(self.Values()):
            mu, sigma = hypo

            # compute log likelihood of m, given hypo
            stderr_m = sigma / math.sqrt(n)
            loglike = EvalNormalLogPdf(m, mu, stderr_m)

            #compute log likelihood of s, given hypo
            stderr_s = sigma / math.sqrt(2 * (n-1))
            loglike += EvalNormalLogPdf(s, sigma, stderr_s)

            self.Incr(hypo, loglike)


def EvalNormalLogPdf(x, mu, sigma):
    """Computes the log PDF of x given mu and sigma.

    x: float values
    mu, sigma: paramemters of Normal

    returns: float log-likelihood
    """
    return scipy.stats.norm.logpdf(x, mu, sigma)


def FindPriorRanges(xs, num_points, num_stderrs=3.0, median_flag=False):
    """Find ranges for mu and sigma with non-negligible likelihood.

    xs: sample
    num_points: number of values in each dimension
    num_stderrs: number of standard errors to include on either side
    
    Returns: sequence of mus, sequence of sigmas    
    """
    def MakeRange(estimate, stderr):
        """Makes a linear range around the estimate.

        estimate: central value
        stderr: standard error of the estimate

        returns: numpy array of float
        """
        spread = stderr * num_stderrs
        array = numpy.linspace(estimate-spread, estimate+spread, num_points)
        return array

    # estimate mean and stddev of xs
    n = len(xs)
    if median_flag:
        m, s = MedianS(xs, num_sigmas=NUM_SIGMAS)
    else:
        m = numpy.mean(xs)
        s = numpy.std(xs)

    print('classical estimators', m, s)

    # compute ranges for m and s
    stderr_m = s / math.sqrt(n)
    mus = MakeRange(m, stderr_m)

    stderr_s = s / math.sqrt(2 * (n-1))
    sigmas = MakeRange(s, stderr_s)

    return mus, sigmas


def Summation(xs, mu, cache={}):
    """Computes the sum of (x-mu)**2 for x in t.

    Caches previous results.

    xs: tuple of values
    mu: hypothetical mean
    cache: cache of previous results
    """
    try:
        return cache[xs, mu]
    except KeyError:
        ds = [(x-mu)**2 for x in xs]
        total = sum(ds)
        cache[xs, mu] = total
        return total


def CoefVariation(suite):
    """Computes the distribution of CV.

    suite: Pmf that maps (x, y) to z

    Returns: Pmf object for CV.
    """
    pmf = thinkbayes2.Pmf()
    for (m, s), p in suite.Items():
        pmf.Incr(s/m, p)
    return pmf


def PlotCdfs(d, labels):
    """Plot CDFs for each sequence in a dictionary.

    Jitters the data and subtracts away the mean.

    d: map from key to sequence of values
    labels: map from key to string label
    """
    thinkplot.Clf()
    for key, xs in d.items():
        mu = thinkbayes2.Mean(xs)
        xs = thinkbayes2.Jitter(xs, 1.3)
        xs = [x-mu for x in xs]
        cdf = thinkbayes2.MakeCdfFromList(xs)
        thinkplot.Cdf(cdf, label=labels[key])
    thinkplot.Show()
                  

def PlotPosterior(suite, pcolor=False, contour=True):
    """Makes a contour plot.
    
    suite: Suite that maps (mu, sigma) to probability
    """
    thinkplot.Clf()
    thinkplot.Contour(suite.GetDict(), pcolor=pcolor, contour=contour)

    thinkplot.Save(root='variability_posterior_%s' % suite.label,
                title='Posterior joint distribution',
                xlabel='Mean height (cm)',
                ylabel='Stddev (cm)')


def PlotCoefVariation(suites):
    """Plot the posterior distributions for CV.

    suites: map from label to Pmf of CVs.
    """
    thinkplot.Clf()
    thinkplot.PrePlot(num=2)

    pmfs = {}
    for label, suite in suites.items():
        pmf = CoefVariation(suite)
        print('CV posterior mean', pmf.Mean())
        cdf = thinkbayes2.MakeCdfFromPmf(pmf, label)
        thinkplot.Cdf(cdf)
    
        pmfs[label] = pmf

    thinkplot.Save(root='variability_cv',
                xlabel='Coefficient of variation',
                ylabel='Probability')

    print('female bigger', thinkbayes2.PmfProbGreater(pmfs['female'],
                                                     pmfs['male']))
    print('male bigger', thinkbayes2.PmfProbGreater(pmfs['male'],
                                                   pmfs['female']))


def PlotOutliers(samples):
    """Make CDFs showing the distribution of outliers."""
    cdfs = []
    for label, sample in samples.items():
        outliers = [x for x in sample if x < 150]

        cdf = thinkbayes2.MakeCdfFromList(outliers, label)
        cdfs.append(cdf)

    thinkplot.Clf()
    thinkplot.Cdfs(cdfs)
    thinkplot.Save(root='variability_cdfs',
                title='CDF of height',
                xlabel='Reported height (cm)',
                ylabel='CDF')


def PlotMarginals(suite):
    """Plots marginal distributions from a joint distribution.

    suite: joint distribution of mu and sigma.
    """
    thinkplot.Clf()

    pyplot.subplot(1, 2, 1)
    pmf_m = suite.Marginal(0)
    cdf_m = thinkbayes2.MakeCdfFromPmf(pmf_m)
    thinkplot.Cdf(cdf_m)

    pyplot.subplot(1, 2, 2)
    pmf_s = suite.Marginal(1)
    cdf_s = thinkbayes2.MakeCdfFromPmf(pmf_s)
    thinkplot.Cdf(cdf_s)

    thinkplot.Show()


def ReadHeights(nrows=None):
    """Read the BRFSS dataset, extract the heights and pickle them.

    nrows: number of rows to read
    """
    resp = brfss.ReadBrfss(nrows=nrows).dropna(subset=['sex', 'htm3'])
    groups = resp.groupby('sex')

    d = {}
    for name, group in groups:
        d[name] = group.htm3.values

    return d


def UpdateSuite1(suite, xs):
    """Computes the posterior distibution of mu and sigma.

    Computes untransformed likelihoods.

    suite: Suite that maps from (mu, sigma) to prob
    xs: sequence
    """
    suite.UpdateSet(xs)


def UpdateSuite2(suite, xs):
    """Computes the posterior distibution of mu and sigma.

    Computes log likelihoods.

    suite: Suite that maps from (mu, sigma) to prob
    xs: sequence
    """
    suite.Log()
    suite.LogUpdateSet(xs)
    suite.Exp()
    suite.Normalize()


def UpdateSuite3(suite, xs):
    """Computes the posterior distibution of mu and sigma.

    Computes log likelihoods efficiently.

    suite: Suite that maps from (mu, sigma) to prob
    t: sequence
    """
    suite.Log()
    suite.LogUpdateSetFast(xs)
    suite.Exp()
    suite.Normalize()


def UpdateSuite4(suite, xs):
    """Computes the posterior distibution of mu and sigma.

    Computes log likelihoods efficiently.

    suite: Suite that maps from (mu, sigma) to prob
    t: sequence
    """
    suite.Log()
    suite.LogUpdateSetMeanVar(xs)
    suite.Exp()
    suite.Normalize()


def UpdateSuite5(suite, xs):
    """Computes the posterior distibution of mu and sigma.

    Computes log likelihoods efficiently.

    suite: Suite that maps from (mu, sigma) to prob
    t: sequence
    """
    suite.Log()
    suite.LogUpdateSetMedianIPR(xs)
    suite.Exp()
    suite.Normalize()


def MedianIPR(xs, p):
    """Computes the median and interpercentile range.

    xs: sequence of values
    p: range (0-1), 0.5 yields the interquartile range

    returns: tuple of float (median, IPR)
    """
    cdf = thinkbayes2.MakeCdfFromList(xs)
    median = cdf.Percentile(50)

    alpha = (1-p) / 2
    ipr = cdf.Value(1-alpha) - cdf.Value(alpha)
    return median, ipr


def MedianS(xs, num_sigmas):
    """Computes the median and an estimate of sigma.

    Based on an interpercentile range (IPR).

    factor: number of standard deviations spanned by the IPR
    """
    half_p = thinkbayes2.StandardNormalCdf(num_sigmas) - 0.5
    median, ipr = MedianIPR(xs, half_p * 2)
    s = ipr / 2 / num_sigmas

    return median, s

def Summarize(xs):
    """Prints summary statistics from a sequence of values.

    xs: sequence of values
    """
    # print smallest and largest
    xs.sort()
    print('smallest', xs[:10])
    print('largest', xs[-10:])

    # print median and interquartile range
    cdf = thinkbayes2.MakeCdfFromList(xs)
    print(cdf.Percentile(25), cdf.Percentile(50), cdf.Percentile(75))


def RunEstimate(update_func, num_points=31, median_flag=False):
    """Runs the whole analysis.

    update_func: which of the update functions to use
    num_points: number of points in the Suite (in each dimension)
    """
    d = ReadHeights(nrows=None)
    labels = {1:'male', 2:'female'}

    # PlotCdfs(d, labels)

    suites = {}
    for key, xs in d.items():
        label = labels[key]
        print(label, len(xs))
        Summarize(xs)

        xs = thinkbayes2.Jitter(xs, 1.3)

        mus, sigmas = FindPriorRanges(xs, num_points, median_flag=median_flag)
        suite = Height(mus, sigmas, label)
        suites[label] = suite
        update_func(suite, xs)
        print('MLE', suite.MaximumLikelihood())

        PlotPosterior(suite)

        pmf_m = suite.Marginal(0)
        pmf_s = suite.Marginal(1)
        print('marginal mu', pmf_m.Mean(), pmf_m.Var())
        print('marginal sigma', pmf_s.Mean(), pmf_s.Var())

        # PlotMarginals(suite)

    PlotCoefVariation(suites)


def main():
    random.seed(17)

    func = UpdateSuite5
    median_flag = (func == UpdateSuite5)
    RunEstimate(func, median_flag=median_flag)


if __name__ == '__main__':
    main()


""" Results:

UpdateSuite1 (100):
marginal mu 162.816901408 0.55779791443
marginal sigma 6.36966103214 0.277026082819

UpdateSuite2 (100):
marginal mu 162.816901408 0.55779791443
marginal sigma 6.36966103214 0.277026082819

UpdateSuite3 (100):
marginal mu 162.816901408 0.55779791443
marginal sigma 6.36966103214 0.277026082819

UpdateSuite4 (100):
marginal mu 162.816901408 0.547456009605
marginal sigma 6.30305516111 0.27544106054

UpdateSuite3 (1000):
marginal mu 163.722137405 0.0660294386397
marginal sigma 6.64453251495 0.0329935312671

UpdateSuite4 (1000):
marginal mu 163.722137405 0.0658920503302
marginal sigma 6.63692197049 0.0329689887609

UpdateSuite3 (all):
marginal mu 163.223475005 0.000203282582659
marginal sigma 7.26918836916 0.000101641131229

UpdateSuite4 (all):
marginal mu 163.223475004 0.000203281499857
marginal sigma 7.26916693422 0.000101640932082

UpdateSuite5 (all):
marginal mu 163.1805214 7.9399898468e-07
marginal sigma 7.29969524118 3.26257030869e-14

"""

