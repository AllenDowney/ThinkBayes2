"""This file contains code used in "Think Bayes",
by Allen B. Downey, available from greenteapress.com

Copyright 2012 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
"""

from __future__ import print_function, division

import math

import thinkbayes2
import thinkplot


class Sat(thinkbayes2.Suite, thinkbayes2.Joint):
    """Represents the distribution of p_correct for a test-taker."""

    def Likelihood(self, data, hypo):
        """Computes the likelihood of data under hypo.

        data: boolean, whether the answer is correct
        hypo: pair of (efficacy, difficulty)
        """
        correct = data
        e, d = hypo
        p = ProbCorrect(e, d)
        like = p if correct else 1-p
        return like


def ProbCorrect(efficacy, difficulty, a=1):
    """Returns the probability that a person gets a question right.

    efficacy: personal ability to answer questions
    difficulty: how hard the question is
    a: parameter that controls the shape of the curve

    Returns: float prob
    """
    return 1 / (1 + math.exp(-a * (efficacy - difficulty)))


def Update(p, q, correct):
    """Updates p and q according to correct.

    p: prior distribution of efficacy for the test-taker
    q: prior distribution of difficulty for the question

    returns: pair of new Pmfs
    """
    joint = thinkbayes2.MakeJoint(p, q)
    suite = Sat(joint)
    suite.Update(correct)
    p, q = suite.Marginal(0, label=p.label), suite.Marginal(1, label=q.label)
    return p, q


def main():
    p1 = thinkbayes2.MakeNormalPmf(0, 1, 3, n=101)
    p1.label = 'p1'
    p2 = p1.Copy(label='p2')

    q1 = thinkbayes2.MakeNormalPmf(0, 1, 3, n=101)
    q1.label = 'q1'
    q2 = q1.Copy(label='q2')

    p1, q1 = Update(p1, q1, True)
    p1, q2 = Update(p1, q2, True)
    p2, q1 = Update(p2, q1, True)
    p2, q2 = Update(p2, q2, False)

    thinkplot.PrePlot(num=4, rows=2)
    thinkplot.Pmfs([p1, p2])
    thinkplot.Config(legend=True)

    thinkplot.SubPlot(2)
    thinkplot.Pmfs([q1, q2])
    thinkplot.Show()

    print('Prob p1 > p2', p1 > p2)
    print('Prob q1 > q2', q1 > q2)


if __name__ == '__main__':
    main()
