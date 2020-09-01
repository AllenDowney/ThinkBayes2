import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from empiricaldist import Pmf


def write_table(table, label, **options):
    """Write a table in LaTex format.
    
    table: DataFrame
    label: string
    options: passed to DataFrame.to_latex
    """
    filename = f'tables/{label}.tex'
    fp = open(filename, 'w')
    s = table.to_latex(**options)
    fp.write(s)
    fp.close()
    
    
def write_pmf(pmf, label):
    """Write a Pmf object as a table.
    
    pmf: Pmf
    label: string
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


def make_uniform(start, stop, num=51, name=None, **options):
    """Make a Pmf that represents a uniform distribution.
    
    start: lower bound
    stop: upper bound
    num: number of points
    name: string name for the quantities
    options: passed to Pmf
    
    returns: Pmf
    """
    qs = np.linspace(start, stop, num)
    pmf = Pmf(1.0, qs, **options)
    pmf.normalize()
    if name:
        pmf.index.name = name
    return pmf


def make_joint(s1, s2):
    """Compute the outer product of two Series.
    
    First Series goes across the columns;
    second goes down the rows.
    
    s1: Series
    s2: Series
    
    return: DataFrame
    """
    X, Y = np.meshgrid(s1, s2)
    return pd.DataFrame(X*Y, columns=s1.index, index=s2.index)


def make_mesh(joint):
    """Make a mesh grid from the quantities in a joint distribution.
    
    joint: DataFrame representing a joint distribution
    
    returns: a mesh grid (X, Y) where X contains the column names and
                                      Y contains the row labels
    """
    x = joint.columns
    y = joint.index
    return np.meshgrid(x, y)


def normalize(joint):
    """Normalize a joint distribution.
    
    joint: DataFrame
    """
    prob_data = joint.to_numpy().sum()
    joint /= prob_data
    return prob_data
    

def marginal(joint, axis):
    """Compute a marginal distribution.
    
    axis=0 returns the marginal distribution of the first variable
    axis=1 returns the marginal distribution of the second variable
    
    joint: DataFrame representing a joint distribution
    axis: int axis to sum along
    
    returns: Pmf
    """
    return Pmf(joint.sum(axis=axis))


def pmf_marginal(joint_pmf, level):
    """Compute a marginal distribution.
    
    joint_pmf: Pmf representing a joint distribution
    level: int, level to sum along
    
    returns: Pmf
    """
    return Pmf(joint_pmf.sum(level=level))


def plot_contour(joint, **options):
    """Plot a joint distribution.
    
    joint: DataFrame representing a joint PMF
    """
    cs = plt.contour(joint.columns, joint.index, joint, **options)
    decorate(xlabel=joint.columns.name, 
             ylabel=joint.index.name)
    return cs


from scipy.stats import binom

def make_binomial(n, p):
    """Make a binomial distribution.
    
    n: number of trials
    p: probability of success
    
    returns: Pmf representing the distribution of k
    """
    ks = np.arange(n+1)
    ps = binom.pmf(ks, n, p)
    return Pmf(ps, ks)


from statsmodels.nonparametric.smoothers_lowess import lowess

def make_lowess(series):
    """Use LOWESS to compute a smooth line.

    series: pd.Series

    returns: pd.Series
    """
    endog = series.values
    exog = series.index.values

    smooth = lowess(endog, exog)
    index, data = np.transpose(smooth)

    return pd.Series(data, index=index)

def plot_series_lowess(series, color):
    """Plots a series of data points and a smooth line.

    series: pd.Series
    color: string or tuple
    """
    series.plot(lw=0, marker='o', color=color, alpha=0.5)
    smooth = make_lowess(series)
    smooth.plot(label='_', color=color)
