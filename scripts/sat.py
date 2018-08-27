"""This file contains code used in "Think Bayes",
by Allen B. Downey, available from greenteapress.com

Copyright 2012 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
"""

from __future__ import print_function, division

import csv
import math
import numpy
import sys

import matplotlib
import matplotlib.pyplot as pyplot

import thinkbayes2
import thinkplot


def ReadScale(filename='sat_scale.csv', col=2):
    """Reads a CSV file of SAT scales (maps from raw score to standard score).

    Args:
      filename: string filename
      col: which column to start with (0=Reading, 2=Math, 4=Writing)

    Returns: thinkbayes2.Interpolator object
    """
    def ParseRange(s):
        """Parse a range of values in the form 123-456

        s: string
        """
        t = [int(x) for x in s.split('-')]
        return 1.0 * sum(t) / len(t)

    fp = open(filename)
    reader = csv.reader(fp)
    raws = []
    scores = []

    for t in reader:
        try:
            raw = int(t[col])
            raws.append(raw)
            score = ParseRange(t[col+1])
            scores.append(score)
        except ValueError:
            pass

    raws.sort()
    scores.sort()
    return thinkbayes2.Interpolator(raws, scores)


def ReadRanks(filename='sat_ranks.csv'):
    """Reads a CSV file of SAT scores.

    Args:
      filename: string filename

    Returns:
      list of (score, freq) pairs
    """
    fp = open(filename)
    reader = csv.reader(fp)
    res = []

    for t in reader:
        try:
            score = int(t[0])
            freq = int(t[1])
            res.append((score, freq))
        except ValueError:
            pass

    return res


def DivideValues(pmf, denom):
    """Divides the values in a Pmf by denom.

    Returns a new Pmf.
    """
    new = thinkbayes2.Pmf()
    denom = float(denom)
    for val, prob in pmf.Items():
        x = val / denom
        new.Set(x, prob)
    return new


class Exam(object):
    """Encapsulates information about an exam.

    Contains the distribution of scaled scores and an
    Interpolator that maps between scaled and raw scores.
    """
    def __init__(self):
        self.scale = ReadScale()

        scores = ReadRanks()
        score_pmf = thinkbayes2.Pmf(dict(scores))

        self.raw = self.ReverseScale(score_pmf)
        self.max_score = max(self.raw.Values())
        self.prior = DivideValues(self.raw, denom=self.max_score)
        
        center = -0.05
        width = 1.8
        self.difficulties = MakeDifficulties(center, width, self.max_score)

    def CompareScores(self, a_score, b_score, constructor):
        """Computes posteriors for two test scores and the likelihood ratio.

        a_score, b_score: scales SAT scores
        constructor: function that instantiates an Sat or Sat2 object
        """
        a_sat = constructor(self, a_score)
        b_sat = constructor(self, b_score)

        a_sat.PlotPosteriors(b_sat)

        if constructor is Sat:
            PlotJointDist(a_sat, b_sat)

        top = TopLevel('AB')
        top.Update((a_sat, b_sat))
        top.Print()

        ratio = top.Prob('A') / top.Prob('B')
        
        print('Likelihood ratio', ratio)

        posterior = ratio / (ratio + 1)
        print('Posterior', posterior)

        if constructor is Sat2:
            ComparePosteriorPredictive(a_sat, b_sat)

    def MakeRawScoreDist(self, efficacies):
        """Makes the distribution of raw scores for given difficulty.

        efficacies: Pmf of efficacy
        """
        pmfs = thinkbayes2.Pmf()
        for efficacy, prob in efficacies.Items():
            scores = self.PmfCorrect(efficacy)
            pmfs.Set(scores, prob)

        mix = thinkbayes2.MakeMixture(pmfs)
        return mix

    def CalibrateDifficulty(self):
        """Make a plot showing the model distribution of raw scores."""
        thinkplot.Clf()
        thinkplot.PrePlot(num=2)

        cdf = thinkbayes2.Cdf(self.raw, label='data')
        thinkplot.Cdf(cdf)

        efficacies = thinkbayes2.MakeNormalPmf(0, 1.5, 3)
        pmf = self.MakeRawScoreDist(efficacies)
        cdf = thinkbayes2.Cdf(pmf, label='model')
        thinkplot.Cdf(cdf)
        
        thinkplot.Save(root='sat_calibrate',
                    xlabel='raw score',
                    ylabel='CDF',
                    formats=['pdf', 'eps'])

    def PmfCorrect(self, efficacy):
        """Returns the PMF of number of correct responses.

        efficacy: float
        """
        pmf = PmfCorrect(efficacy, self.difficulties)
        return pmf

    def Lookup(self, raw):
        """Looks up a raw score and returns a scaled score."""
        return self.scale.Lookup(raw)
        
    def Reverse(self, score):
        """Looks up a scaled score and returns a raw score.

        Since we ignore the penalty, negative scores round up to zero.
        """
        raw = self.scale.Reverse(score)
        return raw if raw > 0 else 0
        
    def ReverseScale(self, pmf):
        """Applies the reverse scale to the values of a PMF.

        Args:
            pmf: Pmf object
            scale: Interpolator object

        Returns:
            new Pmf
        """
        new = thinkbayes2.Pmf()
        for val, prob in pmf.Items():
            raw = self.Reverse(val)
            new.Incr(raw, prob)
        return new


class Sat(thinkbayes2.Suite):
    """Represents the distribution of p_correct for a test-taker."""

    def __init__(self, exam, score):
        self.exam = exam
        self.score = score

        # start with the prior distribution
        thinkbayes2.Suite.__init__(self, exam.prior)

        # update based on an exam score
        self.Update(score)

    def Likelihood(self, data, hypo):
        """Computes the likelihood of a test score, given efficacy."""
        p_correct = hypo
        score = data

        k = self.exam.Reverse(score)
        n = self.exam.max_score
        like = thinkbayes2.EvalBinomialPmf(k, n, p_correct)
        return like

    def PlotPosteriors(self, other):
        """Plots posterior distributions of efficacy.

        self, other: Sat objects.
        """
        thinkplot.Clf()
        thinkplot.PrePlot(num=2)

        cdf1 = thinkbayes2.Cdf(self, label='posterior %d' % self.score)
        cdf2 = thinkbayes2.Cdf(other, label='posterior %d' % other.score)

        thinkplot.Cdfs([cdf1, cdf2])
        thinkplot.Save(xlabel='p_correct', 
                    ylabel='CDF', 
                    axis=[0.7, 1.0, 0.0, 1.0],
                    root='sat_posteriors_p_corr',
                    formats=['pdf', 'eps'])


class Sat2(thinkbayes2.Suite):
    """Represents the distribution of efficacy for a test-taker."""

    def __init__(self, exam, score):
        self.exam = exam
        self.score = score

        # start with the Normal prior
        efficacies = thinkbayes2.MakeNormalPmf(0, 1.5, 3)
        thinkbayes2.Suite.__init__(self, efficacies)

        # update based on an exam score
        self.Update(score)

    def Likelihood(self, data, hypo):
        """Computes the likelihood of a test score, given efficacy."""
        efficacy = hypo
        score = data
        raw = self.exam.Reverse(score)

        pmf = self.exam.PmfCorrect(efficacy)
        like = pmf.Prob(raw)
        return like

    def MakePredictiveDist(self):
        """Returns the distribution of raw scores expected on a re-test."""
        raw_pmf = self.exam.MakeRawScoreDist(self)
        return raw_pmf
    
    def PlotPosteriors(self, other):
        """Plots posterior distributions of efficacy.

        self, other: Sat objects.
        """
        thinkplot.Clf()
        thinkplot.PrePlot(num=2)

        cdf1 = thinkbayes2.Cdf(self, label='posterior %d' % self.score)
        cdf2 = thinkbayes2.Cdf(other, label='posterior %d' % other.score)

        thinkplot.Cdfs([cdf1, cdf2])
        thinkplot.Save(xlabel='efficacy', 
                    ylabel='CDF', 
                    axis=[0, 4.6, 0.0, 1.0],
                    root='sat_posteriors_eff',
                    formats=['pdf', 'eps'])


def PlotJointDist(pmf1, pmf2, thresh=0.8):
    """Plot the joint distribution of p_correct.

    pmf1, pmf2: posterior distributions
    thresh: lower bound of the range to be plotted
    """
    def Clean(pmf):
        """Removes values below thresh."""
        vals = [val for val in pmf.Values() if val < thresh]
        [pmf.Remove(val) for val in vals]

    Clean(pmf1)
    Clean(pmf2)
    pmf = thinkbayes2.MakeJoint(pmf1, pmf2)

    thinkplot.Figure(figsize=(6, 6))    
    thinkplot.Contour(pmf, contour=False, pcolor=True)

    thinkplot.Plot([thresh, 1.0], [thresh, 1.0],
                color='gray', alpha=0.2, linewidth=4)

    thinkplot.Save(root='sat_joint',
                   xlabel='p_correct Alice', 
                   ylabel='p_correct Bob',
                   axis=[thresh, 1.0, thresh, 1.0],
                   formats=['pdf', 'eps'])


def ComparePosteriorPredictive(a_sat, b_sat):
    """Compares the predictive distributions of raw scores.

    a_sat: posterior distribution
    b_sat:
    """
    a_pred = a_sat.MakePredictiveDist()
    b_pred = b_sat.MakePredictiveDist()

    #thinkplot.Clf()
    #thinkplot.Pmfs([a_pred, b_pred])
    #thinkplot.Show()

    a_like = thinkbayes2.PmfProbGreater(a_pred, b_pred)
    b_like = thinkbayes2.PmfProbLess(a_pred, b_pred)
    c_like = thinkbayes2.PmfProbEqual(a_pred, b_pred)

    print('Posterior predictive')
    print('A', a_like)
    print('B', b_like)
    print('C', c_like)


def PlotPriorDist(pmf):
    """Plot the prior distribution of p_correct.

    pmf: prior
    """
    thinkplot.Clf()
    thinkplot.PrePlot(num=1)

    cdf1 = thinkbayes2.Cdf(pmf, label='prior')

    thinkplot.Cdf(cdf1)
    thinkplot.Save(root='sat_prior',
                   xlabel='p_correct', 
                   ylabel='CDF',
                   formats=['pdf', 'eps'])


class TopLevel(thinkbayes2.Suite):
    """Evaluates the top-level hypotheses about Alice and Bob.

    Uses the bottom-level posterior distribution about p_correct
    (or efficacy).
    """

    def Update(self, data):
        a_sat, b_sat = data

        a_like = thinkbayes2.PmfProbGreater(a_sat, b_sat)
        b_like = thinkbayes2.PmfProbLess(a_sat, b_sat)
        c_like = thinkbayes2.PmfProbEqual(a_sat, b_sat)

        a_like += c_like / 2
        b_like += c_like / 2

        self.Mult('A', a_like)
        self.Mult('B', b_like)

        self.Normalize()


def ProbCorrect(efficacy, difficulty, a=1):
    """Returns the probability that a person gets a question right.

    efficacy: personal ability to answer questions
    difficulty: how hard the question is

    Returns: float prob
    """
    return 1 / (1 + math.exp(-a * (efficacy - difficulty)))


def BinaryPmf(p):
    """Makes a Pmf with values 1 and 0.
    
    p: probability given to 1
    
    Returns: Pmf object
    """
    pmf = thinkbayes2.Pmf()
    pmf.Set(1, p)
    pmf.Set(0, 1-p)
    return pmf


def PmfCorrect(efficacy, difficulties):
    """Computes the distribution of correct responses.

    efficacy: personal ability to answer questions
    difficulties: list of difficulties, one for each question

    Returns: new Pmf object
    """
    pmf0 = thinkbayes2.Pmf([0])

    ps = [ProbCorrect(efficacy, difficulty) for difficulty in difficulties]
    pmfs = [BinaryPmf(p) for p in ps]
    dist = sum(pmfs, pmf0)
    return dist


def MakeDifficulties(center, width, n):
    """Makes a list of n difficulties with a given center and width.

    Returns: list of n floats between center-width and center+width
    """
    low, high = center-width, center+width
    return numpy.linspace(low, high, n)


def ProbCorrectTable():
    """Makes a table of p_correct for a range of efficacy and difficulty."""
    efficacies = [3, 1.5, 0, -1.5, -3]
    difficulties = [-1.85, -0.05, 1.75]

    for eff in efficacies:
        print('%0.2f & ' % eff, end=' ') 
        for diff in difficulties:
            p = ProbCorrect(eff, diff)
            print('%0.2f & ' % p, end=' ') 
        print(r'\\')


def main(script):
    ProbCorrectTable()

    exam = Exam()

    PlotPriorDist(exam.prior)
    exam.CalibrateDifficulty()

    exam.CompareScores(780, 740, constructor=Sat)

    exam.CompareScores(780, 740, constructor=Sat2)


if __name__ == '__main__':
    main(*sys.argv)
