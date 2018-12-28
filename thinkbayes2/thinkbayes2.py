"""This file contains code for use with "Think Stats" and
"Think Bayes", both by Allen B. Downey, available from greenteapress.com

Copyright 2018 Allen B. Downey
License: MIT License
"""

import copy
import logging
import math
import random
import re

from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import scipy
from scipy import stats
from scipy import special
from scipy.interpolate import interp1d


from scipy.special import gamma

from io import open


def random_seed(x):
    """Initialize the random and np.random generators.

    x: int seed
    """
    random.seed(x)
    np.random.seed(x)


def odds(p):
    """Computes odds for a given probability.

    Example: p=0.75 means 75 for and 25 against, or 3:1 odds in favor.

    Note: when p=1, the formula for odds divides by zero, which is
    normally undefined.  But I think it is reasonable to define odds(1)
    to be infinity, so that's what this function does.

    p: float 0-1

    Returns: float odds
    """
    if p == 1:
        return float('inf')
    return p / (1 - p)


def probability(o):
    """Computes the probability corresponding to given odds.

    Example: o=2 means 2:1 odds in favor, or 2/3 probability

    o: float odds, strictly positive

    Returns: float probability
    """
    return o / (o + 1)


def probability2(yes, no):
    """Computes the probability corresponding to given odds.

    Example: yes=2, no=1 means 2:1 odds in favor, or 2/3 probability.

    yes, no: int or float odds in favor
    """
    return yes / (yes + no)


# When we plot Hist, Pmf and Cdf objects, they don't appear in
# the legend unless we override the default label.
DEFAULT_LABEL = '_nolegend_'



class Pmf():
    """Represents a probability mass function.

    Values can be any hashable type; probabilities are floating-point.
    Pmfs are not necessarily normalized.
    """

    def Percentile(self, percentage):
        """Computes a percentile of a given Pmf.

        Note: this is not super efficient.  If you are planning
        to compute more than a few percentiles, compute the Cdf.

        percentage: float 0-100

        returns: value from the Pmf
        """
        p = percentage / 100
        total = 0
        for val, prob in sorted(self.items()):
            total += prob
            if total >= p:
                return val

    def Random(self):
        """Chooses a random element from this PMF.

        Note: this is not very efficient.  If you plan to call
        this more than a few times, consider converting to a CDF.

        Returns:
            float value from the Pmf
        """
        target = random.random()
        total = 0
        for x, p in self.d.items():
            total += p
            if total >= target:
                return x

        # we shouldn't get here
        raise ValueError('Random: Pmf might not be normalized.')

    def Sample(self, n):
        """Generates a random sample from this distribution.

        n: int length of the sample
        returns: NumPy array
        """
        return self.MakeCdf().Sample(n)

    def Median(self):
        """Computes the median of a PMF.

        Returns:
            float median
        """
        return self.MakeCdf().Percentile(50)

    def Var(self, mu=None):
        """Computes the variance of a PMF.

        mu: the point around which the variance is computed;
                if omitted, computes the mean

        returns: float variance
        """
        if mu is None:
            mu = self.mean()

        return sum(p * (x-mu)**2 for x, p in self.items())

    def Std(self, mu=None):
        """Computes the standard deviation of a PMF.

        mu: the point around which the variance is computed;
                if omitted, computes the mean

        returns: float standard deviation
        """
        var = self.Var(mu)
        return math.sqrt(var)

    def Mode(self):
        """Returns the value with the highest probability.

        Returns: float probability
        """
        _, val = max((prob, val) for val, prob in self.items())
        return val

    # The mode of a posterior is the maximum aposteori probability (MAP)
    MAP = Mode

    # If the distribution contains likelihoods only, the peak is the
    # maximum likelihood estimator.
    MaximumLikelihood = Mode

    def CredibleInterval(self, percentage=90):
        """Computes the central credible interval.

        If percentage=90, computes the 90% CI.

        Args:
            percentage: float between 0 and 100

        Returns:
            sequence of two floats, low and high
        """
        cdf = self.MakeCdf()
        return cdf.CredibleInterval(percentage)








def MakeJoint(pmf1, pmf2):
    """Joint distribution of values from pmf1 and pmf2.

    Assumes that the PMFs represent independent random variables.

    Args:
        pmf1: Pmf object
        pmf2: Pmf object

    Returns:
        Joint pmf of value pairs
    """
    joint = Joint()
    for v1, p1 in pmf1.items():
        for v2, p2 in pmf2.items():
            joint.Set((v1, v2), p1 * p2)
    return joint


def MakeMixture(metapmf, label='mix'):
    """Make a mixture distribution.

    Args:
      metapmf: Pmf that maps from Pmfs to probs.
      label: string label for the new Pmf.

    Returns: Pmf object.
    """
    mix = Pmf(label=label)
    for pmf, p1 in metapmf.items():
        for x, p2 in pmf.items():
            mix[x] += p1 * p2
    return mix


def MakeUniformPmf(low, high, n):
    """Make a uniform Pmf.

    low: lowest value (inclusive)
    high: highest value (inclusize)
    n: number of values
    """
    pmf = Pmf()
    for x in np.linspace(low, high, n):
        pmf.Set(x, 1)
    pmf.Normalize()
    return pmf





class UnimplementedMethodException(Exception):
    """Exception if someone calls a method that should be overridden."""


class Suite(Pmf):
    """Represents a suite of hypotheses and their probabilities."""

    def Update(self, data):
        """Updates each hypothesis based on the data.

        data: any representation of the data

        returns: the normalizing constant
        """
        for hypo in self.Values():
            like = self.Likelihood(data, hypo)
            self.Mult(hypo, like)
        return self.Normalize()

    def LogUpdate(self, data):
        """Updates a suite of hypotheses based on new data.

        Modifies the suite directly; if you want to keep the original, make
        a copy.

        Note: unlike Update, LogUpdate does not normalize.

        Args:
            data: any representation of the data
        """
        for hypo in self.Values():
            like = self.LogLikelihood(data, hypo)
            self.Incr(hypo, like)

    def UpdateSet(self, dataset):
        """Updates each hypothesis based on the dataset.

        This is more efficient than calling Update repeatedly because
        it waits until the end to Normalize.

        Modifies the suite directly; if you want to keep the original, make
        a copy.

        dataset: a sequence of data

        returns: the normalizing constant
        """
        for data in dataset:
            for hypo in self.Values():
                like = self.Likelihood(data, hypo)
                self.Mult(hypo, like)
        return self.Normalize()

    def LogUpdateSet(self, dataset):
        """Updates each hypothesis based on the dataset.

        Modifies the suite directly; if you want to keep the original, make
        a copy.

        dataset: a sequence of data

        returns: None
        """
        for data in dataset:
            self.LogUpdate(data)

    def Likelihood(self, data, hypo):
        """Computes the likelihood of the data under the hypothesis.

        hypo: some representation of the hypothesis
        data: some representation of the data
        """
        raise UnimplementedMethodException()

    def LogLikelihood(self, data, hypo):
        """Computes the log likelihood of the data under the hypothesis.

        hypo: some representation of the hypothesis
        data: some representation of the data
        """
        raise UnimplementedMethodException()

    def Print(self):
        """Prints the hypotheses and their probabilities."""
        for hypo, prob in sorted(self.items()):
            print(hypo, prob)

    def MakeOdds(self):
        """Transforms from probabilities to odds.

        Values with prob=0 are removed.
        """
        for hypo, prob in self.items():
            if prob:
                self.Set(hypo, odds(prob))
            else:
                self.Remove(hypo)

    def MakeProbs(self):
        """Transforms from odds to probabilities."""
        for hypo, odds in self.items():
            self.Set(hypo, probability(odds))


class Pdf(object):
    """Represents a probability density function (PDF)."""

    def Density(self, x):
        """Evaluates this Pdf at x.

        Returns: float or NumPy array of probability density
        """
        raise UnimplementedMethodException()

    def GetLinspace(self):
        """Get a linspace for plotting.

        Not all subclasses of Pdf implement this.

        Returns: numpy array
        """
        raise UnimplementedMethodException()

    def MakePmf(self, **options):
        """Makes a discrete version of this Pdf.

        options can include
        label: string
        low: low end of range
        high: high end of range
        n: number of places to evaluate

        Returns: new Pmf
        """
        label = options.pop('label', '')
        xs, ds = self.Render(**options)
        return Pmf(dict(zip(xs, ds)), label=label)

    def Render(self, **options):
        """Generates a sequence of points suitable for plotting.

        If options includes low and high, it must also include n;
        in that case the density is evaluated an n locations between
        low and high, including both.

        If options includes xs, the density is evaluate at those location.

        Otherwise, self.GetLinspace is invoked to provide the locations.

        Returns:
            tuple of (xs, densities)
        """
        low, high = options.pop('low', None), options.pop('high', None)
        if low is not None and high is not None:
            n = options.pop('n', 101)
            xs = np.linspace(low, high, n)
        else:
            xs = options.pop('xs', None)
            if xs is None:
                xs = self.GetLinspace()

        ds = self.Density(xs)
        return xs, ds

    def Items(self):
        """Generates a sequence of (value, probability) pairs.
        """
        return zip(*self.Render())


class NormalPdf(Pdf):
    """Represents the PDF of a Normal distribution."""

    def __init__(self, mu=0, sigma=1, label=None):
        """Constructs a Normal Pdf with given mu and sigma.

        mu: mean
        sigma: standard deviation
        label: string
        """
        self.mu = mu
        self.sigma = sigma
        self.label = label if label is not None else '_nolegend_'

    def __str__(self):
        return 'NormalPdf(%f, %f)' % (self.mu, self.sigma)

    def GetLinspace(self):
        """Get a linspace for plotting.

        Returns: numpy array
        """
        low, high = self.mu-3*self.sigma, self.mu+3*self.sigma
        return np.linspace(low, high, 101)

    def Density(self, xs):
        """Evaluates this Pdf at xs.

        xs: scalar or sequence of floats

        returns: float or NumPy array of probability density
        """
        return stats.norm.pdf(xs, self.mu, self.sigma)


class ExponentialPdf(Pdf):
    """Represents the PDF of an exponential distribution."""

    def __init__(self, lam=1, label=None):
        """Constructs an exponential Pdf with given parameter.

        lam: rate parameter
        label: string
        """
        self.lam = lam
        self.label = label if label is not None else '_nolegend_'

    def __str__(self):
        return 'ExponentialPdf(%f)' % (self.lam)

    def GetLinspace(self):
        """Get a linspace for plotting.

        Returns: numpy array
        """
        low, high = 0, 5.0/self.lam
        return np.linspace(low, high, 101)

    def Density(self, xs):
        """Evaluates this Pdf at xs.

        xs: scalar or sequence of floats

        returns: float or NumPy array of probability density
        """
        return stats.expon.pdf(xs, scale=1.0/self.lam)


class EstimatedPdf(Pdf):
    """Represents a PDF estimated by KDE."""

    def __init__(self, sample, label=None):
        """Estimates the density function based on a sample.

        sample: sequence of data
        label: string
        """
        self.label = label if label is not None else '_nolegend_'
        self.kde = stats.gaussian_kde(sample)
        low = min(sample)
        high = max(sample)
        self.linspace = np.linspace(low, high, 101)

    def __str__(self):
        return 'EstimatedPdf(label=%s)' % str(self.label)

    def GetLinspace(self):
        """Get a linspace for plotting.

        Returns: numpy array
        """
        return self.linspace

    def Density(self, xs):
        """Evaluates this Pdf at xs.

        returns: float or NumPy array of probability density
        """
        return self.kde.evaluate(xs)

    def Sample(self, n):
        """Generates a random sample from the estimated Pdf.

        n: size of sample
        """
        # NOTE: we have to flatten because resample returns a 2-D
        # array for some reason.
        return self.kde.resample(n).flatten()


def CredibleInterval(pmf, percentage=90):
    """Computes a credible interval for a given distribution.

    If percentage=90, computes the 90% CI.

    Args:
        pmf: Pmf object representing a posterior distribution
        percentage: float between 0 and 100

    Returns:
        sequence of two floats, low and high
    """
    cdf = pmf.MakeCdf()
    prob = (1 - percentage / 100) / 2
    interval = cdf.Value(prob), cdf.Value(1 - prob)
    return interval



def underride(d, **options):
    """Add key-value pairs to d only if key is not in d.

    d: dictionary
    options: keyword args to add to d
    """
    for key, val in options.items():
        d.setdefault(key, val)

    return d


class Pmf(pd.Series):

    def __init__(self, qs=None, ps=None, name=None, normalize=True):
        """Initializes.

        If ps is provided, qs must be the corresponding list of values.

        qs: Pmf, dict, pandas Series, list of pairs
        ps: sequence of probabilities
        name: string
        normalize: boolean, whether or not to normalize
        """
        if name is None:
            if isinstance(qs, pd.Series):
                name = qs.name

        if qs is None:
            # caller has not provided qs; make an empty Pmf
            super().__init__([], name=name)

            # if the provided ps without qs,
            if ps is not None:
                logging.warning("Pmf: can't pass ps without also passing qs.")
            return
        else:
            # if the caller provides qs and ps
            if ps is not None:
                if isinstance(ps, str):
                    logging.warning("Pmf: ps can't be a string")

                super().__init__(ps, index=qs, name=name)
                return

        # caller has provided just qs, not ps

        # if it's a dict copy it
        if isinstance(qs, dict):
            super().__init__(qs, name=name)
            return

        # if qs is a Cdf
        if isinstance(qs, Pmf):
            super().__init__(qs.values.copy(), index=qs.index.copy(), name=name)
            return

        # if qs is a Cdf
        if isinstance(qs, Cdf):
            diff = np.ediff1d(qs, to_begin=qs.values[0])
            super().__init__(diff, index=qs.index.copy(), name=name)
            return

        # otherwise, treat qs as a sequence of values
        series = pd.Series(qs).value_counts()
        super().__init__(series, name=name)
        if normalize:
            self.normalize()

    def __hash__(self):
        return id(self)

    def __repr__(self):
        cls = self.__class__.__name__
        s = super().__repr__()
        return s

    @property
    def qs(self):
        return self.index.values

    @property
    def ps(self):
        return self.values

    def copy(self):
        return Pmf(self)

    def __call__(self, qs):
        return self.get(qs, 0)

    def __getitem__(self, qs):
        try:
            return super().__getitem__(qs)
        except (KeyError, ValueError, IndexError):
            return 0

    def normalize(self):
        self /= self.sum()

    def sort(self):
        self.sort_index(inplace=True)

    def bar(self, **options):
        underride(options, label=self.name)
        plt.bar(self.index, self.values, **options)

    def plot(self, **options):
        underride(options, label=self.name)
        plt.plot(self.index, self.values, **options)

    def mean(self):
        """Computes the mean of a PMF.

        Returns:
            float mean
        """
        return np.sum(self.qs * self.ps)

    def __add__(self, x):
        """Computes the Pmf of the sum of values drawn from self and other.

        x: another Pmf or a scalar

        returns: new Pmf
        """
        if isinstance(x, Pmf):
            return pmf_add(self, x)
        else:
            return Pmf(self.qs + x, self.ps)

    __radd__ = __add__

    def __sub__(self, x):
        """Computes the Pmf of the diff of values drawn from self and other.

        x: another Pmf

        returns: new Pmf
        """
        if isinstance(x, Pmf):
            return pmf_sub(self, x)
        else:
            return Pmf(self.qs - x, self.ps)

    def marginal(self, i, label=None):
        """Gets the marginal distribution of the indicated variable.

        i: index of the variable we want

        Returns: Pmf
        """
        pmf = Pmf(label=label)
        for vs, p in self.items():
            pmf[vs[i]] += p
        return pmf

    def conditional(self, i, j, val, label=None):
        """Gets the conditional distribution of the indicated variable.

        Distribution of vs[i], conditioned on vs[j] = val.

        i: index of the variable we want
        j: which variable is conditioned on
        val: the value the jth variable has to have

        Returns: Pmf
        """
        pmf = Pmf(label=label)
        for vs, p in self.items():
            if vs[j] == val:
                pmf[vs[i]] += p

        pmf.normalize()
        return pmf

    def max_like_interval(self, percentage=90):
        """Returns the maximum-likelihood credible interval.

        If percentage=90, computes a 90% CI containing the values
        with the highest likelihoods.

        percentage: float between 0 and 100

        Returns: list of values from the suite
        """
        interval = []
        total = 0

        t = [(prob, val) for val, prob in self.items()]
        t.sort(reverse=True)

        for prob, val in t:
            interval.append(val)
            total += prob
            if total >= percentage / 100:
                break

        return interval

    def total(self):
        """Returns the total of the frequencies/probabilities."""
        total = np.sum(self.values)
        return total

    def Largest(self, n=10):
        """Returns the largest n values, with frequency/probability.

        n: number of items to return
        """
        return sorted(self.d.items(), reverse=True)[:n]

    def Smallest(self, n=10):
        """Returns the smallest n values, with frequency/probability.

        n: number of items to return
        """
        return sorted(self.d.items(), reverse=False)[:n]

    def gt(self, x):
        """Probability that a sample from this Pmf > x.

        x: number

        returns: float probability
        """
        if isinstance(x, Pmf):
            return pmf_gt(self, x)
        else:
            return self[self.qs > x].sum()

    def lt(self, x):
        """Probability that a sample from this Pmf < x.

        x: number

        returns: float probability
        """
        if isinstance(x, Pmf):
            return pmf_lt(self, x)
        else:
            return self[self.qs < x].sum()

    def ge(self, x):
        """Probability that a sample from this Pmf >= x.

        x: number

        returns: float probability
        """
        if isinstance(x, Pmf):
            return pmf_ge(self, x)
        else:
            return self[self.qs >= x].sum()

    def le(self, x):
        """Probability that a sample from this Pmf <= x.

        x: number

        returns: float probability
        """
        if isinstance(x, Pmf):
            return pmf_le(self, x)
        else:
            return self[self.qs <= x].sum()

    def eq(self, x):
        """Probability that a sample from this Pmf == x.

        x: number

        returns: float probability
        """
        if isinstance(x, Pmf):
            return pmf_eq(self, x)
        else:
            return self[self.qs == x].sum()

    def ne(self, x):
        """Probability that a sample from this Pmf != x.

        x: number

        returns: float probability
        """
        if isinstance(x, Pmf):
            return pmf_ne(self, x)
        else:
            return self[self.qs != x].sum()


def pmf_conv(pmf1, pmf2, func):
    qs = func.outer(pmf1.qs, pmf2.qs).flatten()
    ps = np.multiply.outer(pmf1.ps, pmf2.ps).flatten()
    series = pd.Series(ps).groupby(qs).sum()
    return Pmf(series.index, series.values)


def pmf_add(pmf1, pmf2):
    return pmf_conv(pmf1, pmf2, np.add)


def pmf_sub(pmf1, pmf2):
    return pmf_conv(pmf1, pmf2, np.subtract)


def pmf_outer(pmf1, pmf2, func):
    """Computes the outer product of two PMFs.

    func: function to apply to the qs

    returns: NumPy array
    """
    qs = func.outer(pmf1.qs, pmf2.qs)
    ps = np.multiply.outer(pmf1.ps, pmf2.ps)
    return qs * ps


def pmf_gt(pmf1, pmf2):
    """Probability that a value from pmf1 is greater than a value from pmf2.

    pmf1: Pmf object
    pmf2: Pmf object

    returns: float probability
    """
    outer = pmf_outer(pmf1, pmf2, np.greater)
    return outer.sum()


def pmf_lt(pmf1, pmf2):
    """Probability that a value from pmf1 is less than a value from pmf2.

    pmf1: Pmf object
    pmf2: Pmf object

    returns: float probability
    """
    outer = pmf_outer(pmf1, pmf2, np.less)
    return outer.sum()


def pmf_ge(pmf1, pmf2):
    """Probability that a value from pmf1 is >= than a value from pmf2.

    pmf1: Pmf object
    pmf2: Pmf object

    returns: float probability
    """
    outer = pmf_outer(pmf1, pmf2, np.greater_equal)
    return outer.sum()


def pmf_le(pmf1, pmf2):
    """Probability that a value from pmf1 is <= than a value from pmf2.

    pmf1: Pmf object
    pmf2: Pmf object

    returns: float probability
    """
    outer = pmf_outer(pmf1, pmf2, np.less_equal)
    return outer.sum()


def pmf_eq(pmf1, pmf2):
    """Probability that a value from pmf1 equals a value from pmf2.

    pmf1: Pmf object
    pmf2: Pmf object

    returns: float probability
    """
    outer = pmf_outer(pmf1, pmf2, np.equal)
    return outer.sum()


def pmf_ne(pmf1, pmf2):
    """Probability that a value from pmf1 is <= than a value from pmf2.

    pmf1: Pmf object
    pmf2: Pmf object

    returns: float probability
    """
    outer = pmf_outer(pmf1, pmf2, np.not_equal)
    return outer.sum()


class interp_wrapper:

    def __init__(self, interp):
        self.interp = interp

    def __getitem__(self, xs):
        return self.interp(xs)

    __call__ = __getitem__


class Cdf(pd.Series):

    def __init__(self, qs=None, ps=None, name=None):
        """Initializes.

        If ps is provided, qs must be the corresponding list of values.

        qs: Pmf, dict, pandas Series, list of pairs
        ps: sequence of probabilities
        name: string
        """
        if name is None:
            if isinstance(qs, pd.Series):
                name = qs.name

        if qs is None:
            # caller has not provided qs; make an empty Pmf
            super().__init__([], name=name)

            # if the provided ps without qs,
            if ps is not None:
                logging.warning("Pmf: can't pass ps without also passing qs.")
            return
        else:
            # if the caller provides qs and ps
            if ps is not None:
                if isinstance(ps, str):
                    logging.warning("Pmf: ps can't be a string")

                super().__init__(ps, index=qs, name=name)
                return

        # caller has provided just qs, not ps

        # if qs is a dict, put the keys and values into a Series and sort it
        if isinstance(qs, dict):
            series = pd.Series(qs)
            series.sort_index(inplace=True)
            super().__init__(series.values, index=series.index, name=name)
            return

        # if qs is a Pmf
        if isinstance(qs, Pmf):
            pmf = qs.sort_index()
            super().__init__(pmf.cumsum(), name=name)
            return

        # if qs is a Cdf
        if isinstance(qs, Cdf):
            super().__init__(qs.values.copy(), index=qs.index.copy(), name=name)
            return

        # otherwise, treat qs as a sequence
        series = pd.Series(qs).value_counts()
        series.sort_index(inplace=True)
        super().__init__(series.cumsum(), name=name)

    def copy(self):
        return Cdf(self)

    @property
    def qs(self):
        return self.index.values

    @property
    def ps(self):
        return self.values

    @property
    def forward(self):
        interp = interp1d(self.qs, self.ps,
                          kind='previous',
                          copy=False,
                          assume_sorted=True,
                          bounds_error=False,
                          fill_value=(0,1))
        return interp_wrapper(interp)

    @property
    def inverse(self):
        interp = interp1d(self.ps, self.qs,
                          kind='next',
                          copy=False,
                          assume_sorted=True,
                          bounds_error=False,
                          fill_value=(self.qs[0], np.nan))
        return interp_wrapper(interp)

    def __getitem__(self, qs):
        return self.forward(qs)

    __call__ = __getitem__

    def percentile_rank(self, qs):
        return self.forward(qs) * 100

    def percentile(self, percentile_ranks):
        return self.inverse(percentile_ranks / 100)

    def sample(self, n):
        """Generates a random sample from this distribution.

        n: int length of the sample
        returns: NumPy array
        """
        ps = np.random.random(n)
        return self.inverse(ps)

    def step(self, **options):
        underride(options, label=self.name, where='post')
        plt.step(self.index, self.values, **options)

    def plot(self, **options):
        underride(options, label=self.name)
        plt.plot(self.index, self.values, **options)


def RandomSum(dists):
    """Chooses a random value from each dist and returns the sum.

    dists: sequence of Pmf or Cdf objects

    returns: numerical sum
    """
    total = sum(dist.Random() for dist in dists)
    return total


def SampleSum(dists, n):
    """Draws a sample of sums from a list of distributions.

    dists: sequence of Pmf or Cdf objects
    n: sample size

    returns: new Pmf of sums
    """
    pmf = Pmf(RandomSum(dists) for i in range(n))
    return pmf


def EvalNormalPdf(x, mu, sigma):
    """Computes the unnormalized PDF of the normal distribution.

    x: value
    mu: mean
    sigma: standard deviation

    returns: float probability density
    """
    return stats.norm.pdf(x, mu, sigma)


def EvalBinomialPmf(k, n, p):
    """Evaluates the binomial PMF.

    k: number of successes
    n: number of trials
    p: probability of success on each trial

    returns: probabily of k successes in n trials with probability p.
    """
    return stats.binom.pmf(k, n, p)


def MakeBinomialPmf(n, p):
    """Evaluates the binomial PMF.

    n: number of trials
    p: probability of success on each trial

    returns: Pmf of number of successes
    """
    pmf = Pmf()
    for k in range(n+1):
        pmf[k] = stats.binom.pmf(k, n, p)
    return pmf


def EvalGammaPdf(x, a):
    """Computes the Gamma PDF.

    x: where to evaluate the PDF
    a: parameter of the gamma distribution

    returns: float probability
    """
    return x**(a-1) * np.exp(-x) / gamma(a)


def MakeGammaPmf(xs, a):
    """Makes a PMF discrete approx to a Gamma distribution.

    lam: parameter lambda in events per unit time
    xs: upper bound of the Pmf

    returns: normalized Pmf
    """
    xs = np.asarray(xs)
    ps = EvalGammaPdf(xs, a)
    pmf = Pmf(dict(zip(xs, ps)))
    pmf.Normalize()
    return pmf


def EvalGeometricPmf(k, p, loc=0):
    """Evaluates the geometric PMF.

    With loc=0: Probability of `k` trials to get one success.
    With loc=-1: Probability of `k` trials before first success.

    k: number of trials
    p: probability of success on each trial
    """
    return stats.geom.pmf(k, p, loc=loc)


def MakeGeometricPmf(p, loc=0, high=10):
    """Evaluates the binomial PMF.

    With loc=0: PMF of trials to get one success.
    With loc=-1: PMF of trials before first success.

    p: probability of success
    high: upper bound where PMF is truncated
    """
    pmf = Pmf()
    for k in range(high):
        pmf[k] = stats.geom.pmf(k, p, loc=loc)
    pmf.Normalize()
    return pmf


def EvalHypergeomPmf(k, N, K, n):
    """Evaluates the hypergeometric PMF.

    Returns the probabily of k successes in n trials from a population
    N with K successes in it.
    """
    return stats.hypergeom.pmf(k, N, K, n)


def EvalPoissonPmf(k, lam):
    """Computes the Poisson PMF.

    k: number of events
    lam: parameter lambda in events per unit time

    returns: float probability
    """
    return stats.poisson.pmf(k, lam)


def MakePoissonPmf(lam, high, step=1):
    """Makes a PMF discrete approx to a Poisson distribution.

    lam: parameter lambda in events per unit time
    high: upper bound of the Pmf

    returns: normalized Pmf
    """
    pmf = Pmf()
    for k in range(0, high + 1, step):
        p = stats.poisson.pmf(k, lam)
        pmf.Set(k, p)
    pmf.Normalize()
    return pmf


def EvalExponentialPdf(x, lam):
    """Computes the exponential PDF.

    x: value
    lam: parameter lambda in events per unit time

    returns: float probability density
    """
    return lam * math.exp(-lam * x)


def EvalExponentialCdf(x, lam):
    """Evaluates CDF of the exponential distribution with parameter lam."""
    return 1 - math.exp(-lam * x)


def MakeExponentialPmf(lam, high, n=200):
    """Makes a PMF discrete approx to an exponential distribution.

    lam: parameter lambda in events per unit time
    high: upper bound
    n: number of values in the Pmf

    returns: normalized Pmf
    """
    pmf = Pmf()
    for x in np.linspace(0, high, n):
        p = EvalExponentialPdf(x, lam)
        pmf.Set(x, p)
    pmf.Normalize()
    return pmf


def EvalWeibullPdf(x, lam, k):
    """Computes the Weibull PDF.

    x: value
    lam: parameter lambda in events per unit time
    k: parameter

    returns: float probability density
    """
    arg = (x / lam)
    return k / lam * arg**(k-1) * np.exp(-arg**k)


def EvalWeibullCdf(x, lam, k):
    """Evaluates CDF of the Weibull distribution."""
    arg = (x / lam)
    return 1 - np.exp(-arg**k)


def MakeWeibullPmf(lam, k, high, n=200):
    """Makes a PMF discrete approx to a Weibull distribution.

    lam: parameter lambda in events per unit time
    k: parameter
    high: upper bound
    n: number of values in the Pmf

    returns: normalized Pmf
    """
    xs = np.linspace(0, high, n)
    ps = EvalWeibullPdf(xs, lam, k)
    ps[np.isinf(ps)] = 0
    return Pmf(dict(zip(xs, ps)))


def EvalParetoPdf(x, xm, alpha):
    """Computes the Pareto.

    xm: minimum value (scale parameter)
    alpha: shape parameter

    returns: float probability density
    """
    return stats.pareto.pdf(x, alpha, scale=xm)


def MakeParetoPmf(xm, alpha, high, num=101):
    """Makes a PMF discrete approx to a Pareto distribution.

    xm: minimum value (scale parameter)
    alpha: shape parameter
    high: upper bound value
    num: number of values

    returns: normalized Pmf
    """
    xs = np.linspace(xm, high, num)
    ps = stats.pareto.pdf(xs, alpha, scale=xm)
    pmf = Pmf(dict(zip(xs, ps)))
    return pmf

def StandardNormalCdf(x):
    """Evaluates the CDF of the standard Normal distribution.

    See http://en.wikipedia.org/wiki/Normal_distribution
    #Cumulative_distribution_function

    Args:
        x: float

    Returns:
        float
    """
    return (math.erf(x / ROOT2) + 1) / 2


def EvalNormalCdf(x, mu=0, sigma=1):
    """Evaluates the CDF of the normal distribution.

    Args:
        x: float

        mu: mean parameter

        sigma: standard deviation parameter

    Returns:
        float
    """
    return stats.norm.cdf(x, loc=mu, scale=sigma)


def EvalNormalCdfInverse(p, mu=0, sigma=1):
    """Evaluates the inverse CDF of the normal distribution.

    See http://en.wikipedia.org/wiki/Normal_distribution#Quantile_function

    Args:
        p: float

        mu: mean parameter

        sigma: standard deviation parameter

    Returns:
        float
    """
    return stats.norm.ppf(p, loc=mu, scale=sigma)


def EvalLognormalPdf(x, mu=0, sigma=1):
    """Evaluates the PDF of the lognormal distribution.

    x: float or sequence
    mu: mean parameter
    sigma: standard deviation parameter

    Returns: float or sequence
    """
    return stats.lognorm.pdf(x, loc=mu, scale=sigma)


def EvalLognormalCdf(x, mu=0, sigma=1):
    """Evaluates the CDF of the lognormal distribution.

    x: float or sequence
    mu: mean parameter
    sigma: standard deviation parameter

    Returns: float or sequence
    """
    return stats.lognorm.cdf(x, loc=mu, scale=sigma)


def RenderExpoCdf(lam, low, high, n=101):
    """Generates sequences of xs and ps for an exponential CDF.

    lam: parameter
    low: float
    high: float
    n: number of points to render

    returns: numpy arrays (xs, ps)
    """
    xs = np.linspace(low, high, n)
    ps = 1 - np.exp(-lam * xs)
    #ps = stats.expon.cdf(xs, scale=1.0/lam)
    return xs, ps


def RenderNormalCdf(mu, sigma, low, high, n=101):
    """Generates sequences of xs and ps for a Normal CDF.

    mu: parameter
    sigma: parameter
    low: float
    high: float
    n: number of points to render

    returns: numpy arrays (xs, ps)
    """
    xs = np.linspace(low, high, n)
    ps = stats.norm.cdf(xs, mu, sigma)
    return xs, ps


def RenderParetoCdf(xmin, alpha, low, high, n=50):
    """Generates sequences of xs and ps for a Pareto CDF.

    xmin: parameter
    alpha: parameter
    low: float
    high: float
    n: number of points to render

    returns: numpy arrays (xs, ps)
    """
    if low < xmin:
        low = xmin
    xs = np.linspace(low, high, n)
    ps = 1 - (xs / xmin) ** -alpha
    #ps = stats.pareto.cdf(xs, scale=xmin, b=alpha)
    return xs, ps


class Beta:
    """Represents a Beta distribution.

    See http://en.wikipedia.org/wiki/Beta_distribution
    """
    def __init__(self, alpha=1, beta=1, label=None):
        """Initializes a Beta distribution."""
        self.alpha = alpha
        self.beta = beta
        self.label = label if label is not None else '_nolegend_'

    def Update(self, data):
        """Updates a Beta distribution.

        data: pair of int (heads, tails)
        """
        heads, tails = data
        self.alpha += heads
        self.beta += tails

    def mean(self):
        """Computes the mean of this distribution."""
        return self.alpha / (self.alpha + self.beta)

    def MAP(self):
        """Computes the value with maximum a posteori probability."""
        a = self.alpha - 1
        b = self.beta - 1
        return a / (a + b)

    def Random(self):
        """Generates a random variate from this distribution."""
        return random.betavariate(self.alpha, self.beta)

    def Sample(self, n):
        """Generates a random sample from this distribution.

        n: int sample size
        """
        size = n,
        return np.random.beta(self.alpha, self.beta, size)

    def EvalPdf(self, x):
        """Evaluates the PDF at x."""
        return x ** (self.alpha - 1) * (1 - x) ** (self.beta - 1)

    def MakePmf(self, steps=101, label=None):
        """Returns a Pmf of this distribution.

        Note: Normally, we just evaluate the PDF at a sequence
        of points and treat the probability density as a probability
        mass.

        But if alpha or beta is less than one, we have to be
        more careful because the PDF goes to infinity at x=0
        and x=1.  In that case we evaluate the CDF and compute
        differences.

        The result is a little funny, because the values at 0 and 1
        are not symmetric.  Nevertheless, it is a reasonable discrete
        model of the continuous distribution, and behaves well as
        the number of values increases.
        """
        if label is None and self.label is not None:
            label = self.label

        if self.alpha < 1 or self.beta < 1:
            cdf = self.MakeCdf()
            pmf = cdf.MakePmf()
            return pmf

        xs = [i / (steps - 1.0) for i in range(steps)]
        probs = [self.EvalPdf(x) for x in xs]
        pmf = Pmf(dict(zip(xs, probs)), label=label)
        return pmf

    def MakeCdf(self, steps=101):
        """Returns the CDF of this distribution."""
        xs = [i / (steps - 1.0) for i in range(steps)]
        ps = special.betainc(self.alpha, self.beta, xs)
        cdf = Cdf(xs, ps)
        return cdf

    def Percentile(self, ps):
        """Returns the given percentiles from this distribution.

        ps: scalar, array, or list of [0-100]
        """
        ps = np.asarray(ps) / 100
        xs = special.betaincinv(self.alpha, self.beta, ps)
        return xs


class Dirichlet(object):
    """Represents a Dirichlet distribution.

    See http://en.wikipedia.org/wiki/Dirichlet_distribution
    """

    def __init__(self, n, conc=1, label=None):
        """Initializes a Dirichlet distribution.

        n: number of dimensions
        conc: concentration parameter (smaller yields more concentration)
        label: string label
        """
        if n < 2:
            raise ValueError('A Dirichlet distribution with '
                             'n<2 makes no sense')

        self.n = n
        self.params = np.ones(n, dtype=np.float) * conc
        self.label = label if label is not None else '_nolegend_'

    def Update(self, data):
        """Updates a Dirichlet distribution.

        data: sequence of observations, in order corresponding to params
        """
        m = len(data)
        self.params[:m] += data

    def Random(self):
        """Generates a random variate from this distribution.

        Returns: normalized vector of fractions
        """
        p = np.random.gamma(self.params)
        return p / p.sum()

    def Likelihood(self, data):
        """Computes the likelihood of the data.

        Selects a random vector of probabilities from this distribution.

        Returns: float probability
        """
        m = len(data)
        if self.n < m:
            return 0

        x = data
        p = self.Random()
        q = p[:m] ** x
        return q.prod()

    def LogLikelihood(self, data):
        """Computes the log likelihood of the data.

        Selects a random vector of probabilities from this distribution.

        Returns: float log probability
        """
        m = len(data)
        if self.n < m:
            return float('-inf')

        x = self.Random()
        y = np.log(x[:m]) * data
        return y.sum()

    def MarginalBeta(self, i):
        """Computes the marginal distribution of the ith element.

        See http://en.wikipedia.org/wiki/Dirichlet_distribution
        #Marginal_distributions

        i: int

        Returns: Beta object
        """
        alpha0 = self.params.sum()
        alpha = self.params[i]
        return Beta(alpha, alpha0 - alpha)

    def PredictivePmf(self, xs, label=None):
        """Makes a predictive distribution.

        xs: values to go into the Pmf

        Returns: Pmf that maps from x to the mean prevalence of x
        """
        alpha0 = self.params.sum()
        ps = self.params / alpha0
        return Pmf(zip(xs, ps), label=label)

    def mean(self):
        """Array of means."""
        return self.params / np.sum(self.params)


def BinomialCoef(n, k):
    """Compute the binomial coefficient "n choose k".

    n: number of trials
    k: number of successes

    Returns: float
    """
    return scipy.special.comb(n, k)


def LogBinomialCoef(n, k):
    """Computes the log of the binomial coefficient.

    http://math.stackexchange.com/questions/64716/
    approximating-the-logarithm-of-the-binomial-coefficient

    n: number of trials
    k: number of successes

    Returns: float
    """
    return n * math.log(n) - k * math.log(k) - (n - k) * math.log(n - k)


def NormalProbability(ys, jitter=0):
    """Generates data for a normal probability plot.

    ys: sequence of values
    jitter: float magnitude of jitter added to the ys

    returns: numpy arrays xs, ys
    """
    n = len(ys)
    xs = np.random.normal(0, 1, n)
    xs.sort()

    if jitter:
        ys = Jitter(ys, jitter)
    else:
        ys = np.array(ys)
    ys.sort()

    return xs, ys


def Jitter(values, jitter=0.5):
    """Jitters the values by adding a uniform variate in (-jitter, jitter).

    values: sequence
    jitter: scalar magnitude of jitter

    returns: new numpy array
    """
    n = len(values)
    return np.random.normal(0, jitter, n) + values


def NormalProbabilityPlot(sample, fit_color='0.8', **options):
    """Makes a normal probability plot with a fitted line.

    sample: sequence of numbers
    fit_color: color string for the fitted line
    options: passed along to Plot
    """
    xs, ys = NormalProbability(sample)
    mean, var = meanVar(sample)
    std = math.sqrt(var)

    fit = FitLine(xs, mean, std)
    plt.plot(*fit, color=fit_color, label='model')

    xs, ys = NormalProbability(sample)
    plt.plot(xs, ys, **options)


class FixedWidthVariables(object):
    """Represents a set of variables in a fixed width file."""

    def __init__(self, variables, index_base=0):
        """Initializes.

        variables: DataFrame
        index_base: are the indices 0 or 1 based?

        Attributes:
        colspecs: list of (start, end) index tuples
        names: list of string variable names
        """
        self.variables = variables

        # note: by default, subtract 1 from colspecs
        self.colspecs = variables[['start', 'end']] - index_base

        # convert colspecs to a list of pair of int
        self.colspecs = self.colspecs.astype(np.int).values.tolist()
        self.names = variables['name']

    def ReadFixedWidth(self, filename, **options):
        """Reads a fixed width ASCII file.

        filename: string filename

        returns: DataFrame
        """
        df = pd.read_fwf(filename,
                             colspecs=self.colspecs,
                             names=self.names,
                             **options)
        return df


def ReadStataDct(dct_file, **options):
    """Reads a Stata dictionary file.

    dct_file: string filename
    options: dict of options passed to open()

    returns: FixedWidthVariables object
    """
    type_map = dict(byte=int, int=int, long=int, float=float,
                    double=float, numeric=float)

    var_info = []
    with open(dct_file, **options) as f:
        for line in f:
            match = re.search( r'_column\(([^)]*)\)', line)
            if not match:
                continue
            start = int(match.group(1))
            t = line.split()
            vtype, name, fstring = t[1:4]
            name = name.lower()
            if vtype.startswith('str'):
                vtype = str
            else:
                vtype = type_map[vtype]
            long_desc = ' '.join(t[4:]).strip('"')
            var_info.append((start, vtype, name, fstring, long_desc))

    columns = ['start', 'type', 'name', 'fstring', 'desc']
    variables = pd.DataFrame(var_info, columns=columns)

    # fill in the end column by shifting the start column
    variables['end'] = variables.start.shift(-1)
    variables.loc[len(variables)-1, 'end'] = 0

    dct = FixedWidthVariables(variables, index_base=1)
    return dct


def Resample(xs, n=None):
    """Draw a sample from xs with the same length as xs.

    xs: sequence
    n: sample size (default: len(xs))

    returns: NumPy array
    """
    if n is None:
        n = len(xs)
    return np.random.choice(xs, n, replace=True)


def SampleRows(df, nrows, replace=False):
    """Choose a sample of rows from a DataFrame.

    df: DataFrame
    nrows: number of rows
    replace: whether to sample with replacement

    returns: DataDf
    """
    indices = np.random.choice(df.index, nrows, replace=replace)
    sample = df.loc[indices]
    return sample


def ResampleRows(df):
    """Resamples rows from a DataFrame.

    df: DataFrame

    returns: DataFrame
    """
    return SampleRows(df, len(df), replace=True)


def ResampleRowsWeighted(df, column='finalwgt'):
    """Resamples a DataFrame using probabilities proportional to given column.

    df: DataFrame
    column: string column name to use as weights

    returns: DataFrame
    """
    weights = df[column].copy()
    weights /= sum(weights)
    indices = np.random.choice(df.index, len(df), replace=True, p=weights)
    sample = df.loc[indices]
    return sample


def PercentileRow(array, p):
    """Selects the row from a sorted array that maps to percentile p.

    p: float 0--100

    returns: NumPy array (one row)
    """
    rows, cols = array.shape
    index = int(rows * p / 100)
    return array[index,]


def PercentileRows(ys_seq, percents):
    """Given a collection of lines, selects percentiles along vertical axis.

    For example, if ys_seq contains simulation results like ys as a
    function of time, and percents contains (5, 95), the result would
    be a 90% CI for each vertical slice of the simulation results.

    ys_seq: sequence of lines (y values)
    percents: list of percentiles (0-100) to select

    returns: list of NumPy arrays, one for each percentile
    """
    nrows = len(ys_seq)
    ncols = len(ys_seq[0])
    array = np.zeros((nrows, ncols))

    for i, ys in enumerate(ys_seq):
        array[i,] = ys

    array = np.sort(array, axis=0)

    rows = [PercentileRow(array, p) for p in percents]
    return rows


class HypothesisTest(object):
    """Represents a hypothesis test."""

    def __init__(self, data):
        """Initializes.

        data: data in whatever form is relevant
        """
        self.data = data
        self.MakeModel()
        self.actual = self.TestStatistic(data)
        self.test_stats = None
        self.test_cdf = None

    def PValue(self, iters=1000):
        """Computes the distribution of the test statistic and p-value.

        iters: number of iterations

        returns: float p-value
        """
        self.test_stats = [self.TestStatistic(self.RunModel())
                           for _ in range(iters)]
        self.test_cdf = Cdf(self.test_stats)

        count = sum(1 for x in self.test_stats if x >= self.actual)
        return count / iters

    def MaxTestStat(self):
        """Returns the largest test statistic seen during simulations.
        """
        return max(self.test_stats)

    def PlotCdf(self, label=None):
        """Draws a Cdf with vertical lines at the observed test stat.
        """
        def VertLine(x):
            """Draws a vertical line at x."""
            plt.plot([x, x], [0, 1], color='0.8')

        VertLine(self.actual)
        self.test_cdf.plot(label=label)

    def TestStatistic(self, data):
        """Computes the test statistic.

        data: data in whatever form is relevant
        """
        raise UnimplementedMethodException()

    def MakeModel(self):
        """Build a model of the null hypothesis.
        """
        pass

    def RunModel(self):
        """Run the model of the null hypothesis.

        returns: simulated data
        """
        raise UnimplementedMethodException()


def main():
    pass


if __name__ == '__main__':
    main()
