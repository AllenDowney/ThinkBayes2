"""This file contains code for use with "Think Bayes",
by Allen B. Downey, available from greenteapress.com

Copyright 2012 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
"""

from __future__ import print_function, division

from thinkbayes2 import Suite


class Monty(Suite):
    def Likelihood(self, data, hypo):
        """Computes the likelihood of the data under the hypothesis.

        hypo: string name of the door where the prize is
        data: string name of the door Monty opened
        """
        if hypo == data:
            return 0
        elif hypo == 'A':
            return 0.5
        else:
            return 1


def main():
    suite = Monty('ABC')
    suite.Update('B')
    suite.Print()


if __name__ == '__main__':
    main()
