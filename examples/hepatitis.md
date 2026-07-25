---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.18.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

You can order print and ebook versions of *Think Bayes 2e* from
[Bookshop.org](https://bookshop.org/a/98697/9781492089469) and
[Amazon](https://amzn.to/334eqGo).


# Tracking an Outbreak with Mark and Recapture

In epidemiology, the true number of people infected in an outbreak is often larger than the number of cases that appear on any official list.
If we have several incomplete lists, and we can match individuals across them, the pattern of overlaps is informative: few repeats suggest many missing cases; many repeats suggest we have seen most of the cases.

The key to this approach is [mark and recapture modeling](https://en.wikipedia.org/wiki/Mark_and_recapture), a method originally developed for ecology, but applicable in many other domains.
In [this excellent paper](https://doi.org/10.1002/9781118445112.stat04855.pub2), 
Anne Chao reviews the approach and presents an example: a hepatitis A outbreak among students at a college in northern Taiwan in 1995.


Three incomplete lists of cases were available:

1. **P-list** (135 cases): serum tests from the Institute of Preventive Medicine of Taiwan.
2. **Q-list** (122 cases): cases reported by local hospitals to the National Quarantine Service.
3. **E-list** (126 cases): questionnaires collected by epidemiologists.

Across the three lists, 271 distinct students were identified.
The question is how many infected students were never listed.


In Chapter 15 of [*Think Bayes*](https://allendowney.github.io/ThinkBayes2/chap15.html), I used this problem as an exercise.
The solution there uses a discrete grid over two parameters: the total number of infections, `N`, and a shared probability, `p`, that any case appears on any given list.
But this simple model is limited and, as we'll see, it systematically underestimates the number of cases.
Then, we'll try a more realistic model and we'll see that the results are better.

[Click here to run this notebook on Colab](https://colab.research.google.com/github/AllenDowney/ThinkBayes2/blob/master/examples/hepatitis.ipynb).

```python tags=["remove-cell"]
# install empiricaldist if necessary
try:
    import empiricaldist
except ImportError:
    !pip install empiricaldist
```

```python tags=["remove-cell"]
# Get utils.py

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
try:
    import pymc as pm
except ImportError:
    !pip install pymc arviz
```

```python tags=["remove-cell"]
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy.stats import norm

import pymc as pm
import arviz as az

from utils import decorate, set_pyplot_params
```

```python tags=["remove-cell"]
np.random.seed(0)
np.set_printoptions(legacy='1.25')
set_pyplot_params()
```

## The Data

We can summarize the data by listing the number of cases that appeared on all three lists, denoted `k111`, the number that appear on each combination of lists, and the number that appear on none of the lists, `k000`. 

```
P  Q  E   count
1  1  1   28
1  1  0   21
1  0  1   17
1  0  0   69
0  1  1   18
0  1  0   55
0  0  1   63
0  0  0   ??
```

Of course `k000` is unknown. That's what we'll estimate!

Here's the data in array form.

```python
# k000 is unknown; the remaining seven cells are observed
observed = np.array([63, 55, 18, 69, 17, 21, 28])
num_seen = observed.sum()
num_seen
```

So 271 distinct cases appear on at least one list.
Once we estimate `k000`, we can compute `N = num_seen + k000`.


## Homogeneous model

We'll start with what I'll call the homogeneous model, because it assumes that every infected person has the same probability of appearing on a given list -- but different probabilities, `p_P`, `p_Q`, and `p_E`, for the three lists.

With that assumption, the probability of each combination, `000` through `111` is the product of combinations of `p_P`, `p_Q`, `p_E` and their complements (see the model below).

For the prior distribution of `k000` we'll use a Poisson distribution with a weakly informative hyperprior on the Poisson mean.
For the list probabilities, the prior is a Beta distribution with parameters (1, 1), which is uniform from 0 to 1.

Here's the model.

```python
coords = {"list": ["P", "Q", "E"]}
```

```python
with pm.Model(coords=coords) as model:
    p = pm.Beta("p", alpha=1, beta=1, dims="list")
    p_P, p_Q, p_E = p[0], p[1], p[2]

    mu = pm.HalfNormal("mu", sigma=100)
    k000 = pm.Poisson("k000", mu=mu)
    N = pm.Deterministic("N", num_seen + k000)

    q_P, q_Q, q_E = 1 - p_P, 1 - p_Q, 1 - p_E
    probs = pm.math.stack(
        [
            q_P * q_Q * q_E,  # 000
            q_P * q_Q * p_E,  # 001
            q_P * p_Q * q_E,  # 010
            q_P * p_Q * p_E,  # 011
            p_P * q_Q * q_E,  # 100
            p_P * q_Q * p_E,  # 101
            p_P * p_Q * q_E,  # 110
            p_P * p_Q * p_E,  # 111
        ]
    )
    counts = pm.math.stack([k000, *observed])

    pm.Potential(
        "likelihood",
        pm.logp(pm.Multinomial.dist(n=N, p=probs), counts),
    )
    
```

As a technical note, you might wonder why we can't write the likelihood like this:

```    
likelihood = pm.Multinomial("likelihood", n=N, p=probs, observed=counts)
```

The problem is that the value we pass as `observed` has to be constant -- it can't be a random variable or contain any random variables -- and `counts` contains the random variable `k000`. 
Instead we use `Potential` to compute the log likelihood of the data directly.


Here's a graphical representation of the model.

```python
pm.model_to_graphviz(model)
```

Notice that `k000` affects the likelihood of the data in two ways: directly as one of the counts and indirectly as a term of `N = num_seen + k000`.

```python
with model:
    idata = pm.sample(2000, tune=2000, random_seed=42, target_accept=0.9)
```

```python
az.summary(idata, var_names=["N", "p", "mu", "k000"])
```

The model samples reasonably well. Here are the posterior distributions.

```python
az.plot_posterior(idata, var_names=["N", "p"])
decorate()
```

And here's the estimated value of `N` with a 90% CI.

```python
az.summary(idata, var_names=["N"], hdi_prob=0.9)
```

The posterior mean, 381, is consistent with Chao’s independence estimator `N̂0` and with the solution in Think Bayes.
Unfortunately, it turns out to be too low. A later campus-wide serum screen put the true number of infections around 545.

The homogeneous model treats each infected student as equally likely to appear on a given list.
It also assumes that, after accounting for the different coverage of the three lists, appearing on one list provides no information about appearing on another.
Both assumptions are questionable.

In reality, some students are easier to detect than others — for example, because they had more severe symptoms or greater contact with medical services.
Such students would be more likely to appear on all three lists, while students with mild or asymptomatic infections might appear on none.

If that's true, the data will contain more repeated observations than the homogeneous model expects.
In that case, the homogeneous model interprets the observed overlap as evidence that the lists have covered most of the population, and it tends to underestimate the number of infections.
And in this example, we can see that it did.

So let's build a better model.


## Individual heterogeneity

We'll assume that each person has a level of detectability, $z_k$, and each list has a baseline detection level $\alpha_P$, $\alpha_Q$, $\alpha_E$.
For a person with detectability $z_k$, the chance they appear on list `j` is `p_{kj}`, where

$$\mathrm{logit}(p_{kj}) = \alpha_j + \sigma z_k$$

Here $\sigma$ controls how much influence individual detectability has, compared to the baseline detection levels.
A person with high $z_k$ is more likely to appear on more than one list; people with low detectability are more likely to be missed entirely.

To implement this model, we don't need a latent variable for every person.
Instead, we'll assume there are 21 detectability levels, `z_k`, and assign each level a weight, `w_k`, that represents the fraction of people in the population at that level.
Then we can compute the cell probabilities using a weighted sum of the cell probabilities conditioned on `z_k`.

$$\pi_{abc} = \sum_k w_k \, P(abc \mid z_k)$$

We'll precompute the values of `z` and the corresponding weights (using the PDF of a normal distribution to make a bell curve).

```python
z = np.linspace(-4, 4, 21)
weights = norm.pdf(z)
weights /= weights.sum()
```

And precompute the pattern of detection/non-detection for each cell.

```python
from itertools import product

patterns = np.array(list(product([0, 1], repeat=3)), dtype=float)
patterns
```

Now here's a function that builds the model.
The argument `sigma_prior_scale` is the scale of the half-normal prior on $\sigma$; we'll vary it later to check sensitivity.

```python
def make_heterogeneity_model(sigma_prior_scale=1):
    with pm.Model(coords=coords) as model:
        alpha = pm.Normal("alpha", mu=0, sigma=1.5, dims="list")
        sigma = pm.HalfNormal("sigma", sigma=sigma_prior_scale)

        # Probability someone with z=0 appears on each list
        p_typical = pm.Deterministic(
            "p_typical", pm.math.sigmoid(alpha), dims="list"
        )

        # Probability someone with each level of z appears on each list
        # shape: level × list
        logits = alpha[None, :] + sigma * z[:, None]
        p_z = pm.math.sigmoid(logits)

        # Population average probability of appearing on each list
        p_avg = pm.Deterministic(
            "p_avg",
            pm.math.sum(p_z * weights[:, None], axis=0),
            dims="list",
        )

        # Probability of each cell for each level of z and list
        # shape: cell × level × list
        y = patterns[:, None, :]
        cell_prob_given_z = pm.math.prod(
            p_z[None, :, :] ** y * (1 - p_z[None, :, :]) ** (1 - y),
            axis=2,
        )

        # Mixture over levels of z; shape: cell
        probs = pm.math.sum(cell_prob_given_z * weights[None, :], axis=1)
        probs = probs / pm.math.sum(probs)

        # The rest of this model is the same as the previous one
        mu = pm.HalfNormal("mu", sigma=100)
        k000 = pm.Poisson("k000", mu=mu)
        N = pm.Deterministic("N", num_seen + k000)

        counts = pm.math.stack([k000, *observed])
        pm.Potential(
            "likelihood",
            pm.logp(pm.Multinomial.dist(n=N, p=probs), counts),
        )
    return model
```

```python
heterogeneity_model = make_heterogeneity_model()
pm.model_to_graphviz(heterogeneity_model)
```

```python
with heterogeneity_model:
    idata_het = pm.sample(2000, tune=2000, random_seed=42)
```

This model doesn't sample as well, but the results are stable enough for our purposes.

```python
az.summary(
    idata_het,
    var_names=["N", "alpha", "sigma", "p_typical", "p_avg", "mu", "k000"],
)
```

```python
az.plot_posterior(idata_het, var_names=["N", "sigma", "p_avg"])
decorate()
```

```python
az.summary(idata_het, var_names=["N"], hdi_prob=0.9)
```

The mean is much closer to the true value, 545.
And the 90% CI contains the true value.

But the price we pay for a more flexible model is a wider CI.
Here are the posterior distributions of the two models (the dashed line is the true value).

```python
az.plot_density(
    [idata, idata_het],
    var_names=["N"],
    data_labels=["Homogeneous", "Heterogeneous"],
)
plt.axvline(545, ls='--', color='gray')
decorate(title="Posterior of N")
```

```python
summary_ind = az.summary(idata, var_names=["N"], hdi_prob=0.9)
summary_het = az.summary(idata_het, var_names=["N"], hdi_prob=0.9)
pd.DataFrame(
    {
        "mean": [summary_ind.loc["N", "mean"], summary_het.loc["N", "mean"]],
        "hdi_5%": [summary_ind.loc["N", "hdi_5%"], summary_het.loc["N", "hdi_5%"]],
        "hdi_95%": [summary_ind.loc["N", "hdi_95%"], summary_het.loc["N", "hdi_95%"]],
    },
    index=["Homogeneous", "Heterogeneous"],
)
```

## Sensitivity to Priors

With only seven observed cells the baselines $\alpha_j$, and $\sigma$ can trade off.
Larger $\sigma$ creates a subgroup with low probability of detection, which can support a larger unseen population.
So the upper tail of `N` may be prior-sensitive.

To check, we'll test a few values of $\sigma_prior_scale$, which controls the width of the prior distribution of `\sigma`.

```python
def fit_heterogeneity(sigma_prior_scale, random_seed=42):
    model = make_heterogeneity_model(sigma_prior_scale)
    with model:
        idata_s = pm.sample(
            1000, tune=1000, random_seed=random_seed, progressbar=False
        )
    return idata_s
```

```python
sensitivity = []
for scale in [0.5, 1.0, 2.0]:
    idata_s = fit_heterogeneity(scale)
    s = az.summary(idata_s, var_names=["N", "sigma"], hdi_prob=0.9)
    sensitivity.append(
        {
            "HalfNormal scale": scale,
            "N mean": s.loc["N", "mean"],
            "N 5%": s.loc["N", "hdi_5%"],
            "N 95%": s.loc["N", "hdi_95%"],
            "sigma mean": s.loc["sigma", "mean"],
        }
    )

pd.DataFrame(sensitivity).round(1)
```

If the prior distribution of `sigma` is too narrow, it restrains the effect of individual variation, which biases the estimate of `N` downward.

But once it's large enough, it doesn't have much effect on the results.

## Discussion

This example demonstrates a feature of Bayesian modeling with PyMC: the ability to develop and test a model incrementally, adding features as needed and comparing successive iterations.

It also demonstrates the impact of AI tools on the development process for Bayesian models. The models in this example are moderately complex. Both of them use a `Potential` to compute a custom likelihood function, and the second involves array operations that might take some time to write, test, and debug. Using various AI tools, I was able to get both models working quickly -- which allowed time to inspect, test, and refine them.
And then I was able to test a few variations to make the models simpler and easier to explain.

I'm not sure I would say it was easy -- this kind of work still requires a fundamental understanding of Bayesian statistics and PyMC. But it is a lot easier than it used to be!

<!-- #region tags=["remove-print"] -->
Copyright 2025 Allen B. Downey

License: [Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
<!-- #endregion -->
