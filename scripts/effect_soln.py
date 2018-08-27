"""This file contains code used in "Think Bayes",
by Allen B. Downey, available from greenteapress.com

Copyright 2014 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
"""

from __future__ import print_function, division

from variability import *

import thinkplot
import thinkbayes2


def RunEstimate(update_func, num_points=31, median_flag=False):
    """Runs the whole analysis.

    update_func: which of the update functions to use
    num_points: number of points in the Suite (in each dimension)
    """
    d = ReadHeights(nrows=None)
    labels = {1:'male', 2:'female'}

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
        print('MAP', suite.MaximumLikelihood())

    suite1 = suites['male']
    suite2 = suites['female']

    mu1 = suite1.Marginal(0)
    sigma1 = suite1.Marginal(1)

    mu2 = suite2.Marginal(0)
    sigma2 = suite2.Marginal(1)

    diff = mu1 - mu2
    sigma = (sigma1 + sigma2) / 2

    pmf_d = diff / sigma

    thinkplot.Cdf(pmf_d.MakeCdf())
    thinkplot.Show(xlabel='# stddev between means',
                   ylabel='PMF')


def main():
    random.seed(17)

    func = UpdateSuite5
    median_flag = (func == UpdateSuite5)
    RunEstimate(func, median_flag=median_flag)


if __name__ == '__main__':
    main()


