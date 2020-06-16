import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from empiricaldist import Pmf


def write_table(table, label, **options):
    """
    """
    filename = f'tables/{label}.tex'
    fp = open(filename, 'w')
    s = table.to_latex(**options)
    fp.write(s)
    fp.close()

def write_pmf(pmf, label):
    """
    """
    df = pd.DataFrame()
    df['qs'] = pmf.index
    df['ps'] = pmf.values
    write_table(df, label, index=False)
    
def underride(d, **options):
    """Add key-value pairs to d only if key is not in d.

    d: dictionary
    options: keyword args to add to d
    """
    for key, val in options.items():
        d.setdefault(key, val)

    return d


def decorate(**options):
    """Decorate the current axes.
    
    Call decorate with keyword arguments like
    decorate(title='Title',
             xlabel='x',
             ylabel='y')
             
    The keyword arguments can be any of the axis properties
    https://matplotlib.org/api/axes_api.html
    """
    ax = plt.gca()
    ax.set(**options)
    
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels)

    plt.tight_layout()
    
    
def savefig(root, **options):
    """Save the current figure.
    
    root: string filename root
    options: passed to plt.savefig
    """
    format = options.pop('format', None)
    if format:
        formats = [format]
    else:
        formats = ['pdf', 'png']

    for format in formats:
        fname = f'figs/{root}.{format}'
        plt.savefig(fname, **options)
        
        
def make_mixture(pmf, pmf_seq):
    """Make a mixture of distributions.
    
    pmf: mapping from each hypothesis to its probability
    pmf_seq: sequence of Pmfs, each representing 
             a conditional distribution for one hypothesis
             
    returns: Pmf representing the mixture
    """
    df = pd.DataFrame(pmf_seq).fillna(0).transpose()
    df *= pmf.ps
    total = df.sum(axis=1)
    return Pmf(total)

def outer_product(s1, s2):
    """Compute the outer product of two Series.
    
    First Series goes down the rows;
    second goes across the columns.
    
    s1: Series
    s2: Series
    
    return: DataFrame
    """
    a = np.multiply.outer(s1.to_numpy(), s2.to_numpy())
    return pd.DataFrame(a, index=s1.index, columns=s2.index)
