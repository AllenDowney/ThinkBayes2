"""This file contains code used in "Think Bayes",
by Allen B. Downey, available from greenteapress.com

Copyright 2013 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
"""

from __future__ import print_function, division

import thinkbayes2

import thinkplot
import numpy

import math
import random
import sys

FORMATS = ['pdf', 'eps', 'png', 'jpg']

"""
Notation guide:

z: time between trains
x: time since the last train
y: time until the next train

zb: distribution of z as seen by a random arrival

"""

# longest hypothetical time between trains, in seconds

UPPER_BOUND = 1200

# observed gaps between trains, in seconds
# collected using code in redline_data.py, run daily 4-6pm
# for 5 days, Monday 6 May 2013 to Friday 10 May 2013

OBSERVED_GAP_TIMES = [
    428.0, 705.0, 407.0, 465.0, 433.0, 425.0, 204.0, 506.0, 143.0, 351.0, 
    450.0, 598.0, 464.0, 749.0, 341.0, 586.0, 754.0, 256.0, 378.0, 435.0, 
    176.0, 405.0, 360.0, 519.0, 648.0, 374.0, 483.0, 537.0, 578.0, 534.0, 
    577.0, 619.0, 538.0, 331.0, 186.0, 629.0, 193.0, 360.0, 660.0, 484.0, 
    512.0, 315.0, 457.0, 404.0, 740.0, 388.0, 357.0, 485.0, 567.0, 160.0, 
    428.0, 387.0, 901.0, 187.0, 622.0, 616.0, 585.0, 474.0, 442.0, 499.0, 
    437.0, 620.0, 351.0, 286.0, 373.0, 232.0, 393.0, 745.0, 636.0, 758.0,
]


def BiasPmf(pmf, label=None, invert=False):
    """Returns the Pmf with oversampling proportional to value.

    If pmf is the distribution of true values, the result is the
    distribution that would be seen if values are oversampled in
    proportion to their values; for example, if you ask students
    how big their classes are, large classes are oversampled in
    proportion to their size.

    If invert=True, computes in inverse operation; for example,
    unbiasing a sample collected from students.

    Args:
      pmf: Pmf object.
      label: string name for the new Pmf.
      invert: boolean

     Returns:
       Pmf object
    """
    new_pmf = pmf.Copy(label=label)

    for x in pmf.Values():
        if invert:
            new_pmf.Mult(x, 1.0/x)
        else:
            new_pmf.Mult(x, x)
        
    new_pmf.Normalize()
    return new_pmf


def UnbiasPmf(pmf, label=None):
    """Returns the Pmf with oversampling proportional to 1/value.

    Args:
      pmf: Pmf object.
      label: string label for the new Pmf.

     Returns:
       Pmf object
    """
    return BiasPmf(pmf, label, invert=True)


def MakeUniformPmf(low, high):
    """Make a uniform Pmf.

    low: lowest value (inclusive)
    high: highest value (inclusive)
    """
    xs = MakeRange(low, high)
    pmf = thinkbayes2.Pmf(xs)
    return pmf    
    

def MakeRange(low=10, high=None, skip=10):
    """Makes a range representing possible gap times in seconds.

    low: where to start
    high: where to end
    skip: how many to skip
    """
    if high is None:
        high = UPPER_BOUND

    xs = numpy.arange(low, high+skip, skip)
    return xs


class WaitTimeCalculator(object):
    """Encapsulates the forward inference process.

    Given the actual distribution of gap times (z),
    computes the distribution of gaps as seen by
    a random passenger (zb), which yields the distribution
    of wait times (y) and the distribution of elapsed times (x).
    """

    def __init__(self, pmf, inverse=False):
        """Constructor.

        pmf: Pmf of either z or zb
        inverse: boolean, true if pmf is zb, false if pmf is z
        """
        if inverse:
            self.pmf_zb = pmf
            self.pmf_z = UnbiasPmf(pmf, label="z")
        else:
            self.pmf_z = pmf
            self.pmf_zb = BiasPmf(pmf, label="zb")

        # distribution of wait time
        self.pmf_y = PmfOfWaitTime(self.pmf_zb)

        # the distribution of elapsed time is the same as the
        # distribution of wait time
        self.pmf_x = self.pmf_y

    def GenerateSampleWaitTimes(self, n):
        """Generates a random sample of wait times.

        n: sample size

        Returns: sequence of values
        """
        cdf_y = thinkbayes2.Cdf(self.pmf_y)
        sample = cdf_y.Sample(n)
        return sample

    def GenerateSampleGaps(self, n):
        """Generates a random sample of gaps seen by passengers.

        n: sample size

        Returns: sequence of values
        """
        cdf_zb = thinkbayes2.Cdf(self.pmf_zb)
        sample = cdf_zb.Sample(n)
        return sample

    def GenerateSamplePassengers(self, lam, n):
        """Generates a sample wait time and number of arrivals.

        lam: arrival rate in passengers per second
        n: number of samples

        Returns: list of (k1, y, k2) tuples
        k1: passengers there on arrival
        y: wait time
        k2: passengers arrived while waiting
        """
        zs = self.GenerateSampleGaps(n)
        xs, ys = SplitGaps(zs)

        res = []
        for x, y in zip(xs, ys):
            k1 = numpy.random.poisson(lam * x)
            k2 = numpy.random.poisson(lam * y)
            res.append((k1, y, k2))

        return res

    def PlotPmfs(self, root='redline0'):
        """Plots the computed Pmfs.

        root: string
        """
        pmfs = ScaleDists([self.pmf_z, self.pmf_zb], 1.0/60)

        thinkplot.Clf()
        thinkplot.PrePlot(2)
        thinkplot.Pmfs(pmfs)
        thinkplot.Save(root=root,
                       xlabel='Time (min)',
                       ylabel='CDF',
                       formats=FORMATS)


    def MakePlot(self, root='redline2'):
        """Plots the computed CDFs.

        root: string
        """
        print('Mean z', self.pmf_z.Mean() / 60)
        print('Mean zb', self.pmf_zb.Mean() / 60)
        print('Mean y', self.pmf_y.Mean() / 60)

        cdf_z = self.pmf_z.MakeCdf()
        cdf_zb = self.pmf_zb.MakeCdf()
        cdf_y = self.pmf_y.MakeCdf()

        cdfs = ScaleDists([cdf_z, cdf_zb, cdf_y], 1.0/60)

        thinkplot.Clf()
        thinkplot.PrePlot(3)
        thinkplot.Cdfs(cdfs)
        thinkplot.Save(root=root,
                       xlabel='Time (min)',
                       ylabel='CDF',
                       formats=FORMATS)


def SplitGaps(zs):
    """Splits zs into xs and ys.

    zs: sequence of gaps

    Returns: tuple of sequences (xs, ys)
    """
    xs = [random.uniform(0, z) for z in zs]
    ys = [z-x for z, x in zip(zs, xs)]
    return xs, ys


def PmfOfWaitTime(pmf_zb):
    """Distribution of wait time.

    pmf_zb: dist of gap time as seen by a random observer

    Returns: dist of wait time (also dist of elapsed time)
    """
    metapmf = thinkbayes2.Pmf()
    for gap, prob in pmf_zb.Items():
        uniform = MakeUniformPmf(0, gap)
        metapmf.Set(uniform, prob)

    pmf_y = thinkbayes2.MakeMixture(metapmf, label='y')
    return pmf_y


def ScaleDists(dists, factor):
    """Scales each of the distributions in a sequence.

    dists: sequence of Pmf or Cdf
    factor: float scale factor
    """
    return [dist.Scale(factor) for dist in dists]


class ElapsedTimeEstimator(object):
    """Uses the number of passengers to estimate time since last train."""

    def __init__(self, wtc, lam, num_passengers):
        """Constructor.

        pmf_x: expected distribution of elapsed time
        lam: arrival rate in passengers per second
        num_passengers: # passengers seen on the platform
        """
        # prior for elapsed time
        self.prior_x = Elapsed(wtc.pmf_x, label='prior x')

        # posterior of elapsed time (based on number of passengers)
        self.post_x = self.prior_x.Copy(label='posterior x')
        self.post_x.Update((lam, num_passengers))

        # predictive distribution of wait time
        self.pmf_y = PredictWaitTime(wtc.pmf_zb, self.post_x)

    def MakePlot(self, root='redline3'):
        """Plot the CDFs.

        root: string
        """
        # observed gaps
        cdf_prior_x = self.prior_x.MakeCdf()
        cdf_post_x = self.post_x.MakeCdf()
        cdf_y = self.pmf_y.MakeCdf()

        cdfs = ScaleDists([cdf_prior_x, cdf_post_x, cdf_y], 1.0/60)

        thinkplot.Clf()
        thinkplot.PrePlot(3)
        thinkplot.Cdfs(cdfs)
        thinkplot.Save(root=root,
                       xlabel='Time (min)',
                       ylabel='CDF',
                       formats=FORMATS)


class ArrivalRate(thinkbayes2.Suite):
    """Represents the distribution of arrival rates (lambda)."""

    def Likelihood(self, data, hypo):
        """Computes the likelihood of the data under the hypothesis.

        Evaluates the Poisson PMF for lambda and k.

        hypo: arrival rate in passengers per second
        data: tuple of elapsed_time and number of passengers
        """
        lam = hypo
        x, k = data
        like = thinkbayes2.EvalPoissonPmf(k, lam * x)
        return like


class ArrivalRateEstimator(object):
    """Estimates arrival rate based on passengers that arrive while waiting.
    """

    def __init__(self, passenger_data):
        """Constructor

        passenger_data: sequence of (k1, y, k2) pairs
        """
        # range for lambda
        low, high = 0, 5
        n = 51
        hypos = numpy.linspace(low, high, n) / 60

        self.prior_lam = ArrivalRate(hypos, label='prior')
        self.prior_lam.Remove(0)

        self.post_lam = self.prior_lam.Copy(label='posterior')

        for _k1, y, k2 in passenger_data:
            self.post_lam.Update((y, k2))

        print('Mean posterior lambda', self.post_lam.Mean())

    def MakePlot(self, root='redline1'):
        """Plot the prior and posterior CDF of passengers arrival rate.

        root: string
        """
        thinkplot.Clf()
        thinkplot.PrePlot(2)

        # convert units to passengers per minute
        prior = self.prior_lam.MakeCdf().Scale(60)
        post = self.post_lam.MakeCdf().Scale(60)

        thinkplot.Cdfs([prior, post])

        thinkplot.Save(root=root,
                       xlabel='Arrival rate (passengers / min)',
                       ylabel='CDF',
                       formats=FORMATS)
                       

class Elapsed(thinkbayes2.Suite):
    """Represents the distribution of elapsed time (x)."""

    def Likelihood(self, data, hypo):
        """Computes the likelihood of the data under the hypothesis.

        Evaluates the Poisson PMF for lambda and k.

        hypo: elapsed time since the last train
        data: tuple of arrival rate and number of passengers
        """
        x = hypo
        lam, k = data
        like = thinkbayes2.EvalPoissonPmf(k, lam * x)
        return like


def PredictWaitTime(pmf_zb, pmf_x):
    """Computes the distribution of wait times.

    Enumerate all pairs of zb from pmf_zb and x from pmf_x,
    and accumulate the distribution of y = z - x.

    pmf_zb: distribution of gaps seen by random observer
    pmf_x: distribution of elapsed time
    """
    pmf_y = pmf_zb - pmf_x
    pmf_y.label = 'pred y'
    RemoveNegatives(pmf_y)
    return pmf_y


def RemoveNegatives(pmf):
    """Removes negative values from a PMF.

    pmf: Pmf
    """
    for val in list(pmf.Values()):
        if val < 0:
            pmf.Remove(val)
    pmf.Normalize()


class Gaps(thinkbayes2.Suite):
    """Represents the distribution of gap times,
    as updated by an observed waiting time."""

    def Likelihood(self, data, hypo):
        """The likelihood of the data under the hypothesis.

        If the actual gap time is z, what is the likelihood
        of waiting y seconds?

        hypo: actual time between trains
        data: observed wait time
        """
        z = hypo
        y = data
        if y > z:
            return 0
        return 1.0 / z


class GapDirichlet(thinkbayes2.Dirichlet):
    """Represents the distribution of prevalences for each
    gap time."""

    def __init__(self, xs):
        """Constructor.

        xs: sequence of possible gap times
        """
        n = len(xs)
        thinkbayes2.Dirichlet.__init__(self, n)
        self.xs = xs
        self.mean_zbs = []

    def PmfMeanZb(self):
        """Makes the Pmf of mean zb.

        Values stored in mean_zbs.
        """
        return thinkbayes2.Pmf(self.mean_zbs)

    def Preload(self, data):
        """Adds pseudocounts to the parameters.

        data: sequence of pseudocounts
        """
        thinkbayes2.Dirichlet.Update(self, data)

    def Update(self, data):
        """Computes the likelihood of the data.

        data: wait time observed by random arrival (y)

        Returns: float probability
        """
        k, y = data

        print(k, y)
        prior = self.PredictivePmf(self.xs)
        gaps = Gaps(prior)
        gaps.Update(y)
        probs = gaps.Probs(self.xs)

        self.params += numpy.array(probs)


class GapDirichlet2(GapDirichlet):
    """Represents the distribution of prevalences for each
    gap time."""

    def Update(self, data):
        """Computes the likelihood of the data.

        data: wait time observed by random arrival (y)

        Returns: float probability
        """
        k, y = data

        # get the current best guess for pmf_z
        pmf_zb = self.PredictivePmf(self.xs)

        # use it to compute prior pmf_x, pmf_y, pmf_z
        wtc = WaitTimeCalculator(pmf_zb, inverse=True)

        # use the observed passengers to estimate posterior pmf_x
        elapsed = ElapsedTimeEstimator(wtc,
                                       lam=0.0333,
                                       num_passengers=k)

        # use posterior_x and observed y to estimate observed z
        obs_zb = elapsed.post_x + Floor(y)
        probs = obs_zb.Probs(self.xs)

        mean_zb = obs_zb.Mean()
        self.mean_zbs.append(mean_zb)
        print(k, y, mean_zb)

        # use observed z to update beliefs about pmf_z
        self.params += numpy.array(probs)


class GapTimeEstimator(object):
    """Infers gap times using passenger data."""

    def __init__(self, xs, pcounts, passenger_data):
        self.xs = xs
        self.pcounts = pcounts
        self.passenger_data = passenger_data

        self.wait_times = [y for _k1, y, _k2 in passenger_data]
        self.pmf_y = thinkbayes2.Pmf(self.wait_times, label="y")

        dirichlet = GapDirichlet2(self.xs)
        dirichlet.params /= 1.0

        dirichlet.Preload(self.pcounts)
        dirichlet.params /= 20.0

        self.prior_zb = dirichlet.PredictivePmf(self.xs, label="prior zb")
        
        for k1, y, _k2 in passenger_data:
            dirichlet.Update((k1, y))

        self.pmf_mean_zb = dirichlet.PmfMeanZb()

        self.post_zb = dirichlet.PredictivePmf(self.xs, label="post zb")
        self.post_z = UnbiasPmf(self.post_zb, label="post z")

    def PlotPmfs(self):
        """Plot the PMFs."""
        print('Mean y', self.pmf_y.Mean())
        print('Mean z', self.post_z.Mean())
        print('Mean zb', self.post_zb.Mean())

        thinkplot.Pmf(self.pmf_y)
        thinkplot.Pmf(self.post_z)
        thinkplot.Pmf(self.post_zb)

    def MakePlot(self):
        """Plot the CDFs."""
        thinkplot.Cdf(self.pmf_y.MakeCdf())
        thinkplot.Cdf(self.prior_zb.MakeCdf())
        thinkplot.Cdf(self.post_zb.MakeCdf())
        thinkplot.Cdf(self.pmf_mean_zb.MakeCdf())
        thinkplot.Show()


def Floor(x, factor=10):
    """Rounds down to the nearest multiple of factor.

    When factor=10, all numbers from 10 to 19 get floored to 10.
    """
    return int(x/factor) * factor


def TestGte():
    """Tests the GapTimeEstimator."""
    random.seed(17)

    xs = [60, 120, 240]
    
    gap_times = [60, 60, 60, 60, 60, 120, 120, 120, 240, 240]

    # distribution of gap time (z)
    pdf_z = thinkbayes2.EstimatedPdf(gap_times)
    pmf_z = pdf_z.MakePmf(xs=xs, label="z")

    wtc = WaitTimeCalculator(pmf_z, inverse=False)

    lam = 0.0333
    n = 100
    passenger_data = wtc.GenerateSamplePassengers(lam, n)

    pcounts = [0, 0, 0]

    ite = GapTimeEstimator(xs, pcounts, passenger_data)

    thinkplot.Clf()

    # thinkplot.Cdf(wtc.pmf_z.MakeCdf(label="actual z"))    
    thinkplot.Cdf(wtc.pmf_zb.MakeCdf(label="actual zb"))
    ite.MakePlot()


class WaitMixtureEstimator(object):
    """Encapsulates the process of estimating wait time with uncertain lam.
    """

    def __init__(self, wtc, are, num_passengers=15):
        """Constructor.

        wtc: WaitTimeCalculator
        are: ArrivalTimeEstimator
        num_passengers: number of passengers seen on the platform
        """
        self.metapmf = thinkbayes2.Pmf()

        for lam, prob in sorted(are.post_lam.Items()):
            ete = ElapsedTimeEstimator(wtc, lam, num_passengers)
            self.metapmf.Set(ete.pmf_y, prob)

        self.mixture = thinkbayes2.MakeMixture(self.metapmf)

        lam = are.post_lam.Mean()
        ete = ElapsedTimeEstimator(wtc, lam, num_passengers)
        self.point = ete.pmf_y

    def MakePlot(self, root='redline4'):
        """Makes a plot showing the mixture."""
        thinkplot.Clf()

        # plot the MetaPmf
        for pmf, prob in sorted(self.metapmf.Items()):
            cdf = pmf.MakeCdf().Scale(1.0/60)
            width = 2/math.log(-math.log(prob))
            thinkplot.Plot(cdf.xs, cdf.ps,
                           alpha=0.2, linewidth=width, color='blue', 
                           label='')

        # plot the mixture and the distribution based on a point estimate
        thinkplot.PrePlot(2)
        #thinkplot.Cdf(self.point.MakeCdf(label='point').Scale(1.0/60))
        thinkplot.Cdf(self.mixture.MakeCdf(label='mix').Scale(1.0/60))

        thinkplot.Save(root=root,
                       xlabel='Wait time (min)',
                       ylabel='CDF',
                       formats=FORMATS,
                       axis=[0,10,0,1])



def GenerateSampleData(gap_times, lam=0.0333, n=10):
    """Generates passenger data based on actual gap times.

    gap_times: sequence of float
    lam: arrival rate in passengers per second
    n: number of simulated observations
    """
    xs = MakeRange(low=10)
    pdf_z = thinkbayes2.EstimatedPdf(gap_times)
    pmf_z = pdf_z.MakePmf(xs=xs, label="z")

    wtc = WaitTimeCalculator(pmf_z, inverse=False)
    passenger_data = wtc.GenerateSamplePassengers(lam, n)
    return wtc, passenger_data


def RandomSeed(x):
    """Initialize the random and numpy.random generators.

    x: int seed
    """
    random.seed(x)
    numpy.random.seed(x)
    

def RunSimpleProcess(gap_times, lam=0.0333, num_passengers=15, plot=True):
    """Runs the basic analysis and generates figures.

    gap_times: sequence of float
    lam: arrival rate in passengers per second
    num_passengers: int number of passengers on the platform
    plot: boolean, whether to generate plots

    Returns: WaitTimeCalculator, ElapsedTimeEstimator
    """
    global UPPER_BOUND
    UPPER_BOUND = 1200

    cdf_z = thinkbayes2.Cdf(gap_times).Scale(1.0/60)
    print('CI z', cdf_z.CredibleInterval(90))

    xs = MakeRange(low=10)

    pdf_z = thinkbayes2.EstimatedPdf(gap_times)
    pmf_z = pdf_z.MakePmf(xs=xs, label="z")

    wtc = WaitTimeCalculator(pmf_z, inverse=False)    

    if plot:
        wtc.PlotPmfs()
        wtc.MakePlot()

    ete = ElapsedTimeEstimator(wtc, lam, num_passengers)

    if plot:
        ete.MakePlot()

    return wtc, ete


def RunMixProcess(gap_times, lam=0.0333, num_passengers=15, plot=True):
    """Runs the analysis for unknown lambda.

    gap_times: sequence of float
    lam: arrival rate in passengers per second
    num_passengers: int number of passengers on the platform
    plot: boolean, whether to generate plots

    Returns: WaitMixtureEstimator
    """
    global UPPER_BOUND
    UPPER_BOUND = 1200

    wtc, _ete = RunSimpleProcess(gap_times, lam, num_passengers)

    RandomSeed(20)
    passenger_data = wtc.GenerateSamplePassengers(lam, n=5)

    total_y = 0
    total_k2 = 0
    for k1, y, k2 in passenger_data:
        print(k1, y/60, k2)
        total_y += y/60
        total_k2 += k2
    print(total_k2, total_y)
    print('Average arrival rate', total_k2 / total_y)

    are = ArrivalRateEstimator(passenger_data)

    if plot:
        are.MakePlot()

    wme = WaitMixtureEstimator(wtc, are, num_passengers)

    if plot:
        wme.MakePlot()

    return wme


def RunLoop(gap_times, nums, lam=0.0333):
    """Runs the basic analysis for a range of num_passengers.

    gap_times: sequence of float
    nums: sequence of values for num_passengers
    lam: arrival rate in passengers per second

    Returns: WaitMixtureEstimator
    """
    global UPPER_BOUND
    UPPER_BOUND = 4000

    thinkplot.Clf()

    RandomSeed(18)

    # resample gap_times
    n = 220
    cdf_z = thinkbayes2.Cdf(gap_times)
    sample_z = cdf_z.Sample(n)
    pmf_z = thinkbayes2.Pmf(sample_z)

    # compute the biased pmf and add some long delays
    cdf_zp = BiasPmf(pmf_z).MakeCdf()
    sample_zb = numpy.append(cdf_zp.Sample(n), [1800, 2400, 3000])

    # smooth the distribution of zb
    pdf_zb = thinkbayes2.EstimatedPdf(sample_zb)
    xs = MakeRange(low=60)
    pmf_zb = pdf_zb.MakePmf(xs=xs)

    # unbias the distribution of zb and make wtc
    pmf_z = UnbiasPmf(pmf_zb)
    wtc = WaitTimeCalculator(pmf_z)

    probs = []
    for num_passengers in nums:
        ete = ElapsedTimeEstimator(wtc, lam, num_passengers)

        # compute the posterior prob of waiting more than 15 minutes
        cdf_y = ete.pmf_y.MakeCdf()
        prob = 1 - cdf_y.Prob(900)
        probs.append(prob)

        # thinkplot.Cdf(ete.pmf_y.MakeCdf(label=str(num_passengers)))
    
    thinkplot.Plot(nums, probs)
    thinkplot.Save(root='redline5',
                   xlabel='Num passengers',
                   ylabel='P(y > 15 min)',
                   formats=FORMATS,
                   )


def main(script):
    RunLoop(OBSERVED_GAP_TIMES, nums=[0, 5, 10, 15, 20, 25, 30, 35])
    RunMixProcess(OBSERVED_GAP_TIMES)
    

if __name__ == '__main__':
    main(*sys.argv)
