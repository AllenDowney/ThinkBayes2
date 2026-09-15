---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.5
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

You can order print and ebook versions of *Think Bayes 2e* from
[Bookshop.org](https://bookshop.org/a/98697/9781492089469) and
[Amazon](https://amzn.to/334eqGo).


# Grid algorithms for hierarchical models


It is widely believed that grid algorithms are only practical for models with 1-3 parameters, or maybe 4-5 if you are careful.
[I've said so myself](https://allendowney.github.io/ThinkBayes2/chap19.html).

But recently I used a grid algorithm to solve the [emitter-detector problem](https://www.allendowney.com/blog/2021/09/05/emitter-detector-redux/), and along the way I noticed something about the structure of the problem: although the model has two parameters, the data only depend on one of them.
That makes it possible to evaluate the likelihood function and update the model very efficiently.

Many hierarchical models have a similar structure: the data depend on a small number of parameters, which depend on a small number of hyperparameters.
I wondered whether the same method would generalize to more complex models, and it does.

As an example, in this notebook I'll use a logitnormal-binomial hierarchical model to solve a problem with two hyperparameters and 13 parameters.
The grid algorithm is not just practical; it's substantially faster than MCMC.

```python tags=["remove-cell"]
# If we're running on Colab, install libraries
import sys
IN_COLAB = 'google.colab' in sys.modules

if IN_COLAB:
    !pip install --quiet --upgrade "pymc>=6" "arviz>=1" "matplotlib>=3.11" empiricaldist
```

```python tags=["remove-cell"]
# Get utils.py, which provides plot_contour

from os.path import basename, exists

def download(url):
    filename = basename(url)
    if not exists(filename):
        from urllib.request import urlretrieve
        local, _ = urlretrieve(url, filename)
        print('Downloaded ' + local)

download('https://github.com/AllenDowney/ThinkBayes2/raw/master/soln/utils.py')
```

```python tags=["remove-cell"]
# PyMC generates a FutureWarning we don't need to deal with

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
```

The following are some utility functions I'll use.

```python
import matplotlib.pyplot as plt

def legend(**options):
    """Make a legend only if there are labels."""
    handles, labels = plt.gca().get_legend_handles_labels()
    if len(labels):
        plt.legend(**options)
```

```python
def decorate(**options):
    plt.gca().set(**options)
    legend()
    plt.tight_layout()
```

```python
from empiricaldist import Cdf

def compare_cdf(pmf, sample):
    pmf.make_cdf().step(label='grid')
    Cdf.from_seq(sample).plot(label='mcmc')
    print(f'grid {pmf.mean():.4f}, mcmc {float(sample.mean()):.4f}')
    decorate()
```

```python
from empiricaldist import Pmf

def make_pmf(ps, qs, name):
    pmf = Pmf(ps, qs)
    pmf.normalize()
    pmf.index.name = name
    return pmf
```

## Heart Attack Data

The problem I'll solve is based on [Chapter 10 of *Probability and Bayesian Modeling*](https://bayesball.github.io/BOOK/bayesian-hierarchical-modeling.html#example-deaths-after-heart-attack); it uses data on death rates due to heart attack for patients treated at various hospitals in New York City.

We can use Pandas to read the data into a `DataFrame`.

```python
import os

filename = 'DeathHeartAttackManhattan.csv'
if not os.path.exists(filename):
    !wget https://github.com/AllenDowney/BayesianInferencePyMC/raw/main/DeathHeartAttackManhattan.csv
```

```python
import pandas as pd

df = pd.read_csv(filename)
df
```

The columns we need are `Cases`, which is the number of patients treated at each hospital, and `Deaths`, which is the number of those patients who died.

```python
data_ns = df['Cases'].values
data_ks = df['Deaths'].values
```

## Solution with PyMC

Here's a hierarchical model that estimates the death rate for each hospital and simultaneously estimates the distribution of rates across hospitals.

`LogitNormal` is supported on the interval from 0 to 1, but PyMC does not
assign it a transform automatically, so the sampler would propose values
outside that range and the model would fail to initialize.
Passing `default_transform=logodds` puts `xs` on an unconstrained scale, as
PyMC does automatically for `sigma`.

```python
import pymc as pm
from pymc.distributions.transforms import logodds

def make_model():
    with pm.Model() as model:
        mu = pm.Normal('mu', 0, 2)
        sigma = pm.HalfNormal('sigma', sigma=1)
        xs = pm.LogitNormal('xs', mu=mu, sigma=sigma, shape=len(data_ns),
                            default_transform=logodds)
        ks = pm.Binomial('ks', n=data_ns, p=xs, observed=data_ks)
    return model
```

```python
%time model = make_model()
pm.model_to_graphviz(model)
```

```python
with model:
    pred = pm.sample_prior_predictive(1000, random_seed=42)
    %time trace = pm.sample(500, target_accept=0.97, random_seed=42)
```

To be fair, PyMC doesn't like this parameterization much (although I'm not sure why). On most runs, there are a moderate number of divergences. Even so, the results are good enough. 

PyMC returns a `DataTree` with the samples indexed by chain and draw.
`az.extract` stacks those two dimensions into a single `sample` dimension, which is more convenient here.

```python
import arviz as az

post = az.extract(trace)
prior_sample = az.extract(pred, group='prior')
```

Here are the posterior distributions of the hyperparameters.

```python
az.plot_dist(trace, var_names=['mu', 'sigma'])
```

And we can extract the posterior distributions of the xs, with one row per hospital.

```python
trace_xs = post['xs']
trace_xs.shape
```

As an example, here's the posterior distribution of x for the first hospital.

```python
Cdf.from_seq(trace_xs[0]).plot()
decorate(title='Posterior distribution of x for the first hospital',
         xlabel='Death rate', ylabel='CDF')
```

## The grid priors

Now let's solve the same problem using a grid algorithm.
I'll use the same priors for the hyperparameters, approximated by a grid with about 100 elements in each dimension.

```python
import numpy as np
from scipy.stats import norm

mus = np.linspace(-6, 6, 101)
ps = norm.pdf(mus, 0, 2)
prior_mu = make_pmf(ps, mus, 'mu')

prior_mu.plot()
decorate(title='Prior distribution of mu')
```

```python
from scipy.stats import logistic

sigmas = np.linspace(0.03, 3.6, 90)
ps = norm.pdf(sigmas, 0, 1)
prior_sigma = make_pmf(ps, sigmas, 'sigma')

prior_sigma.plot()
decorate(title='Prior distribution of sigma')
```

The following cells confirm that these priors are consistent with the prior samples from PyMC.

```python
compare_cdf(prior_mu, prior_sample['mu'])
decorate(title='Prior distribution of mu')
```

```python
compare_cdf(prior_sigma, prior_sample['sigma'])
decorate(title='Prior distribution of sigma')
```


## The joint distribution of hyperparameters

I'll use `make_joint` to make an array that represents the joint prior distribution of the hyperparameters.

```python
def make_joint(prior_x, prior_y):
    X, Y = np.meshgrid(prior_x.ps, prior_y.ps, indexing='ij')
    hyper = X * Y
    return hyper
```

```python
prior_hyper = make_joint(prior_mu, prior_sigma)
prior_hyper.shape
```

Here's what it looks like.

```python
import pandas as pd
from utils import plot_contour

plot_contour(pd.DataFrame(prior_hyper, index=mus, columns=sigmas))
decorate(title="Joint prior of mu and sigma")
```

## Joint prior of hyperparameters and x

Now we're ready to lay out the grid for x, which is the proportion we'll estimate for each hospital.

```python
xs = np.linspace(0.01, 0.99, 295)
```

For each pair of hyperparameters, we'll compute the distribution of `x`.

If `x` has a logit-normal distribution, `logit(x)` has a normal distribution.
But the grid is laid out in equal steps of `x`, not `logit(x)`, so to get a density with respect to `x` we have to include the derivative of the transformation,

$$\frac{d}{dx} \mathrm{logit}(x) = \frac{1}{x (1-x)}$$

Without this factor, the distribution we compute is not logit-normal, and the error is larger when `sigma` is larger.

```python
from scipy.special import logit

M, S, X = np.meshgrid(mus, sigmas, xs, indexing='ij')
LO = logit(X)
jacobian = 1 / (X * (1-X))
LO.sum()
```

```python
from scipy.stats import norm

%time normpdf = norm.pdf(LO, M, S) * jacobian
normpdf.sum()
```


We can speed this up by skipping the terms that don't depend on x

```python
%%time

z = (LO-M) / S
normpdf = np.exp(-z**2/2) * jacobian
```

The result is a 3-D array with axes for mu, sigma, and x.

Now we need to normalize each distribution of `x`.

```python
totals = normpdf.sum(axis=2)
totals.shape
```

To normalize, we have to use a safe version of `divide` where `0/0` is `0`.

```python
def divide(x, y):
    out = np.zeros_like(x)
    return np.divide(x, y, out=out, where=(y!=0))    
```

```python
shape = totals.shape + (1,)
normpdf = divide(normpdf, totals.reshape(shape))
normpdf.shape
```

The result is an array that contains the distribution of `x` for each pair of hyperparameters.

Now, to get the prior distribution, we multiply through by the joint distribution of the hyperparameters.

```python
def make_prior(hyper):

    # reshape hyper so we can multiply along axis 0
    shape = hyper.shape + (1,)
    prior = normpdf * hyper.reshape(shape)

    return prior
```

```python
%time prior = make_prior(prior_hyper)
prior.sum()
```

The result is a 3-D array that represents the joint prior distribution of `mu`, `sigma`, and `x`.

To check that it is correct, I'll extract the marginal distributions and compare them to the priors.

```python
def marginal(joint, axis):
    axes = [i for i in range(3) if i != axis]
    total = joint.sum(axis=tuple(axes))
    return total / total.sum()
```

```python
prior_mu.plot()
marginal_mu = Pmf(marginal(prior, 0), mus)
marginal_mu.plot()
decorate(title='Checking the marginal distribution of mu')
```

```python
prior_sigma.plot()
marginal_sigma = Pmf(marginal(prior, 1), sigmas)
marginal_sigma.plot()
decorate(title='Checking the marginal distribution of sigma')
```

We didn't compute the prior distribution of `x` explicitly; it follows from the distribution of the hyperparameters. But we can extract the prior marginal of `x` from the joint prior.

```python
marginal_x = Pmf(marginal(prior, 2), xs)
marginal_x.plot()
decorate(title='Checking the marginal distribution of x',
         ylim=[0, np.max(marginal_x) * 1.05])
```

And compare it to the prior sample from PyMC.

```python
pred_xs = prior_sample['xs']
pred_xs.shape
```

```python
compare_cdf(marginal_x, pred_xs[0])
decorate(title='Prior distribution of x')
```

The distributions agree, which confirms that the grid represents the same prior as the PyMC model.

An earlier version of this notebook left out the Jacobian factor, and the prior from the grid was noticeably different from the prior from PyMC.
It made little difference to the posteriors in this example, because the posterior distribution of `sigma` is small and the factor is nearly constant over a narrow range of `x`.
But it is wrong in general, and the discrepancy grows with `sigma`.

In addition to the marginals, we'll also find it useful to extract the joint marginal distribution of the hyperparameters.

```python
def get_hyper(joint):
    return joint.sum(axis=2)
```

```python
hyper = get_hyper(prior)
```

```python
plot_contour(pd.DataFrame(hyper, 
                          index=mus, 
                          columns=sigmas))
decorate(title="Joint prior of mu and sigma")
```

## The Update

The likelihood of the data only depends on `x`, so we can compute it like this.

```python
from scipy.stats import binom

data_k = data_ks[0]
data_n = data_ns[0]

like_x = binom.pmf(data_k, data_n, xs)
like_x.shape
```

```python
plt.plot(xs, like_x)
decorate(title='Likelihood of the data')
```

And here's the update.

```python
def update(prior, data):
    n, k = data
    like_x = binom.pmf(k, n, xs)
    posterior = prior * like_x
    posterior /= posterior.sum()
    return posterior
```

```python
data = data_n, data_k
%time posterior = update(prior, data)
```

## Serial updates

At this point we can do an update based on a single hospital, but how do we update based on all of the hospitals?

As a step toward the right answer, I'll start with a wrong answer, which is to do the updates one at a time.

After each update, we extract the posterior distribution of the hyperparameters and use it to create the prior for the next update.

At the end, the posterior distribution of hyperparameters is correct, and the marginal posterior of `x` for the *last* hospital is correct, but the other marginals are wrong because they do not take into account data from subsequent hospitals.

```python
def multiple_updates(prior, ns, ks):
    for data in zip(ns, ks):
        print(data)
        posterior = update(prior, data)
        hyper = get_hyper(posterior)
        prior = make_prior(hyper)
    return posterior
```

```python
%time posterior = multiple_updates(prior, data_ns, data_ks)
```

Here are the posterior distributions of the hyperparameters, compared to the results from PyMC.

```python
marginal_mu = Pmf(marginal(posterior, 0), mus)
compare_cdf(marginal_mu, post['mu'])
```

```python
marginal_sigma = Pmf(marginal(posterior, 1), sigmas)
compare_cdf(marginal_sigma, post['sigma'])
```

```python
marginal_x = Pmf(marginal(posterior, 2), xs)
compare_cdf(marginal_x, trace_xs[-1])
```

## Parallel updates

Doing updates one at time is not quite right, but it gives us an insight.

Suppose we start with a uniform distribution for the hyperparameters and do an update with data from one hospital. If we extract the posterior joint distribution of the hyperparameters, what we get is the likelihood function associated with one dataset.

The following function computes these likelihood functions and saves them in an array called `hyper_likelihood`.

```python
def compute_hyper_likelihood(ns, ks):
    shape = ns.shape + mus.shape + sigmas.shape
    hyper_likelihood = np.empty(shape)
    
    for i, data in enumerate(zip(ns, ks)):
        print(data)
        n, k = data
        like_x = binom.pmf(k, n, xs)
        posterior = normpdf * like_x
        hyper_likelihood[i] = get_hyper(posterior)
    return hyper_likelihood
```

```python
%time hyper_likelihood = compute_hyper_likelihood(data_ns, data_ks)
```

We can multiply this out to get the product of the likelihoods.

```python
%time hyper_likelihood_all = hyper_likelihood.prod(axis=0)
hyper_likelihood_all.sum()
```

This is useful because it provides an efficient way to compute the marginal posterior distribution of `x` for any hospital.
Here's an example.

```python
i = 3
data = data_ns[i], data_ks[i]
data
```

Suppose we did the updates serially and saved this hospital for last.
The prior distribution for the final update would reflect the updates from all previous hospitals, which we can compute by dividing out `hyper_likelihood[i]`.

```python
%time hyper_i = divide(prior_hyper * hyper_likelihood_all, hyper_likelihood[i])
hyper_i.sum()
```

We can use `hyper_i` to make the prior for the last update.

```python
prior_i = make_prior(hyper_i) 
```

And then do the update.

```python
posterior_i = update(prior_i, data)
```

And we can confirm that the results are similar to the results from PyMC.

```python
marginal_mu = Pmf(marginal(posterior_i, 0), mus)
marginal_sigma = Pmf(marginal(posterior_i, 1), sigmas)
marginal_x = Pmf(marginal(posterior_i, 2), xs)
```

```python
compare_cdf(marginal_mu, post['mu'])
```

```python
compare_cdf(marginal_sigma, post['sigma'])
```

```python
compare_cdf(marginal_x, trace_xs[i])
```

## Compute all marginals

The following function computes the marginals for all hospitals and stores the results in an array.

```python
def compute_all_marginals(ns, ks):
    shape = len(ns), len(xs)
    marginal_xs = np.zeros(shape)
    numerator = prior_hyper * hyper_likelihood_all
    
    for i, data in enumerate(zip(ns, ks)):
        hyper_i = divide(numerator, hyper_likelihood[i])
        prior_i = make_prior(hyper_i) 
        posterior_i = update(prior_i, data)
        marginal_xs[i] = marginal(posterior_i, 2)
        
    return marginal_xs
```

```python
%time marginal_xs = compute_all_marginals(data_ns, data_ks)
```

Here's what the results look like, compared to the results from PyMC.

```python
for i, ps in enumerate(marginal_xs):
    pmf = Pmf(ps, xs)
    plt.figure()
    compare_cdf(pmf, trace_xs[i])
    decorate(title=f'Posterior marginal of x for Hospital {i}',
             xlabel='Death rate',
             ylabel='CDF',
             xlim=[trace_xs[i].min(), trace_xs[i].max()])
```

And here are the percentage differences between the results from the grid algorithm and PyMC. All of them are less than 1%.

```python
for i, ps in enumerate(marginal_xs):
    pmf = Pmf(ps, xs)
    diff = abs(pmf.mean() - float(trace_xs[i].mean())) / pmf.mean()
    print(f'{diff * 100:.2f}%')
```

The total time to do all of these computations is about 300 ms, compared to more than 10 seconds to make and run the PyMC model. And PyMC used 4 cores; I only used one.

The grid algorithm is easy to parallelize, and it's incremental. If you get data from a new hospital, or new data for an existing one, you can:

1) Compute the posterior distribution of `x` for the updated hospital, using existing `hyper_likelihoods` for the other hospitals.

2) Update `hyper_likelihoods` for the other hospitals, and run their updates again.

The total time would be about half of what it takes to start from scratch, and it's easy to parallelize.

One drawback of the grid algorithm is that it generates marginal distributions for each hospital rather than a sample from the joint distribution of all of them. So it's less easy to see the correlations among them.

The other drawback, in general, is that it takes more work to set up the grid algorithm. If we switch to another parameterization, it's easier to change the PyMC model.

<!-- #region tags=["remove-print"] -->
Copyright 2021 Allen B. Downey

License: [Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
<!-- #endregion -->
