"""This file contains code for use with "Think Bayes",
by Allen B. Downey, available from greenteapress.com

Copyright 2012 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
"""

from __future__ import print_function, division

"""This file contains a partial solution to a problem from
MacKay, "Information Theory, Inference, and Learning Algorithms."

    Exercise 3.15 (page 50): A statistical statement appeared in
    "The Guardian" on Friday January 4, 2002:

        When spun on edge 250 times, a Belgian one-euro coin came
        up heads 140 times and tails 110.  'It looks very suspicious
        to me,' said Barry Blight, a statistics lecturer at the London
        School of Economics.  'If the coin were unbiased, the chance of
        getting a result as extreme as that would be less than 7%.'

MacKay asks, "But do these data give evidence that the coin is biased
rather than fair?"

"""

import thinkbayes2


class Euro(thinkbayes2.Suite):
    """Represents hypotheses about the probability of heads."""

    def Likelihood(self, data, hypo):
        """Computes the likelihood of the data under the hypothesis.

        hypo: integer value of x, the probability of heads (0-100)
        data: tuple of (number of heads, number of tails)
        """
        x = hypo / 100.0
        heads, tails = data
        like = x**heads * (1-x)**tails
        return like


def TrianglePrior():
    """Makes a Suite with a triangular prior."""
    suite = Euro()
    for x in range(0, 51):
        suite.Set(x, x)
    for x in range(51, 101):
        suite.Set(x, 100-x) 
    suite.Normalize()
    return suite


def SuiteLikelihood(suite, data):
    """Computes the weighted average of likelihoods for sub-hypotheses.

    suite: Suite that maps sub-hypotheses to probability
    data: some representation of the data
   
    returns: float likelihood
    """
    total = 0
    for hypo, prob in suite.Items():
        like = suite.Likelihood(data, hypo)
        total += prob * like
    return total


def Main():
    data = 140, 110
    data = 8, 12

    suite = Euro()
    like_f = suite.Likelihood(data, 50)
    print('p(D|F)', like_f)

    actual_percent = 100.0 * 140 / 250
    likelihood = suite.Likelihood(data, actual_percent)
    print('p(D|B_cheat)', likelihood)
    print('p(D|B_cheat) / p(D|F)', likelihood / like_f)

    like40 = suite.Likelihood(data, 40)
    like60 = suite.Likelihood(data, 60)
    likelihood = 0.5 * like40 + 0.5 * like60
    print('p(D|B_two)', likelihood)
    print('p(D|B_two) / p(D|F)', likelihood / like_f)

    b_uniform = Euro(range(0, 101))
    b_uniform.Remove(50)
    b_uniform.Normalize()
    likelihood = SuiteLikelihood(b_uniform, data)
    print('p(D|B_uniform)', likelihood)
    print('p(D|B_uniform) / p(D|F)', likelihood / like_f)

    b_tri = TrianglePrior()
    b_tri.Remove(50)
    b_tri.Normalize()
    likelihood = b_tri.Update(data)
    print('p(D|B_tri)', likelihood)
    print('p(D|B_tri) / p(D|F)', likelihood / like_f)


if __name__ == '__main__':
    Main()
