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


# Cancer Survival Rates Are Misleading

Five-year survival might be the most misleading statistic in medicine.
For example, suppose 5-year survival for a hypothetical cancer is 

* 91% among patients diagnosed early, while the tumor is **localized** at the primary site,

* 74% among patients diagnosed later, when the tumor has spread **regionally** to nearby lymph nodes or adjacent organs, and

* 16% among patients diagnosed late, when the tumor has spread to **distant** organs or lymph nodes.

What can we infer from these statistics?

1. If a patient is diagnosed early, it is tempting to think the probability is 91% that they will survive five years after diagnosis.

2. Looking at the difference in survival between early and late detection, it is tempting to conclude that more screening would save lives.

3. In a case where a patient is diagnosed late and dies of cancer, it is tempting to say that they would have survived if their cancer had been caught early.

4. And if 5-year survival increases over time, it is tempting to conclude that treatment has improved.

In fact, **none of these inferences are correct**.

Let's take them one at a time.


## Particularization

Here's the first incorrect inference:

> If a patient is diagnosed early, and 5-year survival is 91%, it is tempting to think the probability is 91% that they will survive five years after diagnosis.

This is *almost* correct in the sense that it applies to the past cases that were used to estimate the survival rate -- of all patients in the dataset who were diagnosed early, 91% of them survived at least five years.

But it is misleading for two reasons:

* Because it is based on past cases, it doesn't apply to present cases if (1) the effectiveness of treatment has changed or -- often more importantly -- (2) diagnostic practices have changed.

* Also, before interpreting a probability like this, which applies in general, it is important to **particularize** it for a specific case.

Factors that should be taken into account include the general health of the patient, their age, and the mode of detection.
Some factors are **causal** -- for example, general health directly improves the chance of survival.
Other factors are less obvious because they are **informational** -- for example, the mode of detection can make a big difference:

* If a cancer is discovered because it is causing symptoms, it is more likely to be larger, more aggressive, and relatively late for a given stage -- and all of those implications decrease the chance of survival.

* If a cancer is not symptomatic, but discovered during a physical exam, it is probably larger, later, and more likely to cause mortality, compared to one discovered by high resolution imaging or a sensitive chemical test.

* Conversely, tumors detected by screening are more likely to be slow-growing because of [length-biased sampling](https://en.wikipedia.org/wiki/Length_time_bias) -- the probability of detection depends on the time between when a tumor is detectable and when it causes symptoms.

Taking age into account is complicated because it might be both causal and informational, with opposite implications.
A young patient might be more robust and able to tolerate treatment, but a cancer detectable in a younger person is likely to have progressed more quickly than one that could only be discovered after more years of life.
So the implication of age might be negative among the youngest and oldest patients, and positive in the middle-aged.

For some cancers, the magnitude of these implications is large, so the probability of 5-year survival for a particular patient might be higher than 91% or much lower.


## Is More Screening Better?

Now let's consider the second incorrect inference.

> If 5-year survival is high when a cancer is detected early and much lower when it is detected late, it is tempting to conclude that more screening would save lives.

For example, in a [recent video](https://www.youtube.com/watch?v=ph2ZeNooFLg), Nassim Taleb and Emi Gal discuss the pros and cons of cancer screening, especially full-body MRIs for people who have no symptoms.
At one point they consider this table of survival rates based on stage at diagnosis:

<img width="600" src="https://github.com/AllenDowney/ThinkBayes2/raw/master/images/cancer_table.png">

They note that survival is highest if a tumor is detected while localized at the primary site, lower if it has spread regionally, and often much lower if it has spread distantly.


They take this as evidence that screening for these cancers is beneficial.
For example, [at one point](https://youtu.be/ph2ZeNooFLg?t=357) Taleb says, "Look at the payoff for pancreatic cancer -- 10 times the survival rate."

And Gal adds, "Colon cancer, it's like seven times... The overarching insight is that you want to find cancer early... This table makes the case for the importance of finding cancer early."

Taleb agrees, but this inference is incorrect: **This table does not make the case that it is better to catch cancer early**.


Catching cancer early is beneficial only if (1) the cancers we catch would otherwise cause disease and death, *and* (2) we have treatments that prevent those outcomes, *and* (3) these benefits outweigh the costs of additional screening.
This table does not show that any of those things is true.

In fact, it is possible for a cancer to reproduce any row in this table, even if we have no treatment and detection has no effect on outcomes.
To demonstrate, I'll use a model of tumor progression to show that a hypothetical cancer could have the same survival rates as colon cancer -- **even if there is no effective treatment at all**.

To be clear, I'm not saying that cancer treatment is not effective -- in many cases we know that it is.
I'm saying that we can't tell, just looking at a survival rates, whether early detection has any benefit at all.


[The details of the model are here](https://allendowney.github.io/ThinkBayes2/cancer.html) and you can [click here to run my analysis in a Jupyter notebook on Colab](https://colab.research.google.com/github/AllenDowney/ThinkBayes2/blob/master/examples/cancer.ipynb).

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
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from utils import decorate, set_pyplot_params

np.random.seed(0)
np.set_printoptions(legacy='1.25')
set_pyplot_params()
```

## Data

I downloaded 5-year survival data from [SEER 17](https://canques.seer.cancer.gov/cq_results.php?dir=surv2021&db=101&rpt=TAB&sel=1,2^1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36^0^0^0^1,2,3,4^5&y=Stage%20at%20diagnosis^1,2,3,4&x=Site^1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36&z=Statistic%20type^1,2&dec=1,0,1&template=null), including diagnoses from 2014-2020, male and female, all races, all ages at diagnosis.

```python
tables = pd.read_html('CanQues Results.html')
```

```python
def clean_table(df):
    df = df.set_index("Unnamed: 0")
    df.index.name = 'Site'
    df = df.replace({'%': ''}, regex=True).apply(pd.to_numeric, errors='coerce')
    return df
```

Here are the first few rows of the 5-year survival table.

```python
df1 = clean_table(tables[0])
df1.head()
```

In this data, 5-year survival for colon cancer is 91.4% if it's detected when localized, 74% if it is not detected until it has spread regionally, and 15.8% if it has spread distantly.
For this example, I'll assume that the unknown/unstaged cancers are a mixture of the other categories and we'll leave them out of the analysis.

The dataset also includes the number of cases diagnosed at each stage.

```python
df2 = clean_table(tables[1])
df2.head()
```

I'll normalize the rows so we have the percentage of cancers diagnosed at each stage.

```python
cols = ['Localized', 'Regional', 'Distant']
subset = df2[cols]
incidence = subset.div(subset.sum(axis=1), axis=0) * 100
incidence.head()
```

Here are the statistics we'll try to replicate in the model:

* Five-year survival: 91% if localized / 74% if regional / 16% if distant

* Distribution of stage at diagnosis: 38% localized / 38% regional / 24% distant

Note that there are only five degrees of freedom because the second row has to add up to 100%.


## Markov Model

We'll model tumor progression using a Markov chain with these states:

* `U1`, `U2`, and `U3` represent tumors that are undetected at each stage: local, regional, and distant.

* `D1`, `D2`, `D3` represent tumors that were detected/diagnosed at each stage.

* And `M` represents mortality.

The following function builds the model.

```python
import networkx as nx

def make_graph(lams, kappas, mu, gamma=0):
    G = nx.DiGraph()

    for i, lam in enumerate(lams):
        G.add_edge(f'U{i+1}', f'U{i+2}', p=lam)
        G.add_edge(f'D{i+1}', f'D{i+2}', p=(1 - gamma) * lam)

    for i, kappa in enumerate(kappas):
        G.add_edge(f'U{i+1}', f'D{i+1}', p=kappa)

    
    G.add_edge('D3', 'M', p=mu)
    return G
```

The transition probabilities are:

* `lams`, two values that represent transition rates between stages,

* `kappas`, three values that represent detection rates at each state,

* `mu` the mortality rate from `D3`,

* `gamma` the effectiveness of treatment.

If `gamma > 0`, the treatment is effective by decreasing the probability of progression to the next stage.
In the models we'll run for this example, `gamma=0`, which means that detection has no effect on progression.
In reality, `gamma=0` could mean either:

* There is no effective treatment, or

* The benefit of treatment is offset by the negative implication of detection.

To explain the second point, detection might be more likely if a cancer causes symptoms, and a cancer that causes symptoms might be more likely to progress, compared to an asymptomatic cancer at the same stage.

So, in the absence of effective treatment, we might expect `gamma < 0`.
A moderately effective treatment would have to overcome this effect to bring `gamma` up to 0.

Here are the values of these parameters I chose to fit the data.

```python
lams = [0.15, 0.16]
kappas = [0.09, 0.18, 0.80]
mu = 0.3

G = make_graph(lams, kappas, mu, gamma=0)
```

Overall, these values are within the range we observe in real cancers.

* The transition rates are close to 1/6, which means the average time at each stage is 6 simulated years.

* The detection rate is low at the first stage, higher at the second, and much higher at the third.

* The mortality rate is close to 1/3, so the average survival after diagnosis at the third stage is about 3 years.


The following figure shows the states and transition rates of the model.

```python
edge_labels = {(u, v): f"{d['p']:.2f}" 
               for u, v, d in G.edges(data=True)}
pos = {}

for i, state in enumerate(['U1', 'U2', 'U3']):
    pos[state] = (i, 1)

for i, state in enumerate(['D1', 'D2', 'D3']):
    pos[state] = (i, 0)

pos['M'] = (3, 0)

nx.draw_networkx(G, pos=pos, node_color='lightblue', node_size=500)
                             
nx.draw_networkx_edge_labels(G, pos, 
                             edge_labels=edge_labels,
                             font_size=11,
                             font_color='C0')
plt.axis('equal')
plt.tight_layout()
plt.savefig('cancer1.png', dpi=150)
```

In this model, death occurs only after a cancer has progressed to the third stage and been detected.
That's not realistic -- in reality deaths can occur at any stage, due to cancer or other causes.

But adding more transitions would not make make the model better.
The purpose of the model is to show that we can reproduce the survival rates we see in reality, even if there are no effective treatments.
Making the model more realistic would increase the number of parameters, which would make it easier to reproduce the data, but that would not make the conclusion stronger.
More realistic models are not necessarily better.

But if this feature of the model still bothers you, it might help to think of the states more abstractly.
For example, a transition from `D2` to `D3` might literally represent a cancer that progresses from one stage to another, but it could also include someone whose risk of mortality has increased to be comparable to someone at the next stage, due to other causes.


## Simulation

To simulate the model, we'll use the Markov chain implementation in the [QuantEcon package](https://quanteconpy.readthedocs.io/en/latest/)).

The following function extracts the probabilities in the graph and makes a transition matrix we can use with QuantEcon.

```python
def make_transition_matrix(G, states):
    n_states = len(states)

    # Create a mapping from state names to indices
    state_to_idx = {state: i for i, state in enumerate(states)}

    # Initialize the transition matrix
    P = np.eye(n_states)

    # Fill in the transition probabilities from the graph
    for i, state in enumerate(states):

        for neighbor in G.successors(state):
            j = state_to_idx[neighbor]
            prob = G[state][neighbor]['p']
            P[i, j] = prob
            P[i, i] -= prob

    return P
```

```python
states = ['U1', 'U2', 'U3', 'D1', 'D2', 'D3', 'M']
P = make_transition_matrix(G, states)
pd.DataFrame(P, index=states, columns=states)
```

Now we can make a `MarkovChain` object.

```python
from quantecon import MarkovChain

mc = MarkovChain(P, state_values=states)
```

We'll run 100 simulated years so we have survival times for all cases (with high probability).

```python
state_seq = mc.simulate(ts_length=100, init='U1')
state_seq
```

To analyze the results, we'll use the following function, which takes a state sequence and returns the time of diagnosis and stage at diagnosis.

```python
def diagnosed(state_seq):
    for i, state in enumerate(state_seq):
        if state.startswith('D'):
            return i, state
    return None, None
```

```python
diagnosis_index, stage = diagnosed(state_seq)
diagnosis_index, stage
```

And we'll use the following function to check 5-year survival after diagnosis.
In the rare case where we reach the end of the sequence, that case is counted as a survival.

```python
def survived(state_seq, diagnosis_index, duration=5):
    idx = diagnosis_index + duration
    if idx < len(state_seq):
        return state_seq[idx] != 'M'
    else:
        return True
```

```python
survived(state_seq, diagnosis_index)
```

Now we'll run many simulations and collect the results.
If a case is never diagnosed, it is omitted -- but that never happens in this example.

```python
n_simulations = 10000
results = []

for _ in range(n_simulations):
    state_seq = mc.simulate(ts_length=100, init='U1')
    diagnosis_index, stage = diagnosed(state_seq)
    if diagnosis_index is not None:
        flag = survived(state_seq, diagnosis_index)
        results.append((diagnosis_index, stage, flag))
        
len(results)
```

Here are the first few rows.

```python
df = pd.DataFrame(results, columns=['age', 'stage', 'survived'])
df.head()
```

For the hypothetical cancer in the model, overall 5-year survival is about 64%, the same as colon cancer in reality.

```python
df['survived'].mean()
```

Here are the survival rates by stage.

```python
rates = df.groupby('stage')['survived'].mean() * 100
pd.DataFrame(dict(rates=rates))
```

In the model, survival rates are 95% if localized, 72% if spread regionally, and 17% if spread distantly.
For colon cancer, the rates from the SEER data are are 91%, 74%, and 16%.
So the simulation results are not exactly the same, but they are close.

And here is the distribution of stage at diagnosis.

```python
from empiricaldist import Pmf

Pmf.from_seq(df['stage'], name='incidence') * 100
```

In the model, 38% of tumors are localized when diagnosed, 33% have spread regionally, and 29% have spread distantly.
The actual distribution for colon cancer is 38%, 38%, and 24%.
Again, the simulation results are not exactly the same, but close.

With more trial and error, I could probably find parameters that reproduce the results exactly.
That would not be surprising, because even though the model is meant to be parsimonious, it has seven parameters and we are matching observations with only five degrees of freedom.

That might seem unfair, but it makes the point that there is not enough information in the survival table -- even if we also consider the distribution of stages -- to estimate the parameters of the model precisely.
The data don't exclude the possibility that treatment is ineffective, so they don't prove that early detection is beneficial.

Again, it *might* be better to find cancer early, if the benefit of treatment outweighs the costs of false discovery and overdiagnosis -- but that's a different analysis, and 5-year survival rates aren't part of it.

That's why [this editorial](https://pmc.ncbi.nlm.nih.gov/articles/PMC2733256/) concludes, "Only reduced mortality rates can prove that screening saves lives... journal editors should no longer allow misleading statistics such as five year survival to be reported as evidence for screening."


## Counterfactuals

Now let's consider the third incorrect inference.

> In a case where a patient is diagnosed late and dies of cancer, it is tempting to say that they would have survived if their cancer had been caught early.

For example, [later in the previous video](https://youtu.be/ph2ZeNooFLg?t=836), Taleb says, "Your mother, had she had a colonoscopy, she would be alive today... she's no longer with us because it was detected when it was stage IV, right?" And Gal agrees.

That *might* be true, if treatment would have prevented the cancer from progressing.
But this conclusion is not supported by the data in the survival table.

If someone is diagnosed late and dies, it is tempting to look at the survival table and think there's a 91% chance they would have survived if they had been diagnosed earlier.
But that's not accurate -- and it might not even be close.


First, remember what 91% survival means: among people diagnosed early, 91% survived five years after diagnosis.
But among those survivors, an unknown proportion had tumors that would not have been fatal, even without treatment.
Some might be non-progressive, or progress so slowly that they never cause disease or death.
But in a case where the patient dies of cancer, we know their tumor was not one of those.

As a simplified example, suppose that of all tumors that are caught early, 50% would cause death within five years, if untreated, and 50% would not.
Now imagine 100 people, all detected early and all treated: 50 would survive with or without treatment; out of the other 50, 41 survive with treatment -- so overall survival is 91%.
But if we know someone is in the second group, their chance of survival is not 41/50, which is 82%.

And if the percentage of non-progressive cancers is higher than 50%, the survival rate for progressive cancers is even lower, holding overall survival constant.
So that's one reason the inference is incorrect.


To see the other reason, let's be precise about the counterfactual scenario.
Suppose someone was diagnosed in 2020 with a tumor that had spread distantly, and they died in 2022.
Would they be alive in 2025 if they had been diagnosed earlier?

That depends on when the hypothetical diagnosis happens -- if we imagine they were diagnosed in 2020, five year survival *might* apply (except for the previous point).
But if it had spread distantly in 2020, we have to go farther back in time to catch it early.
For example, if it took 10 years to progress, catching it early means catching it in 2010.
In that case, being "alive today" would depend on 15-year survival, not five.

The five-year survival rate answers the question, "Of all people diagnosed at stage I, how many survive five years?" That is a straightforward statistic to compute.

But the hypothetical asks a different question: "Of all people who died [during a particular interval] after being diagnosed late, how many would be alive [at some later point] if the tumor had been detected early?" That is a *much* harder question to answer -- and five-year survival provides little or no help.


In general, we don't know the probability that someone would be alive today, if they had been diagnosed earlier.
Among other things, it depends on progression rates with and without treatment.

* If many of the tumors caught early would not have progressed or caused death, even without treatment, the counterfactual probability would be low. 

* In any case, if treatment is ineffective, as in the hypothetical cancer we simulated, the counterfactual probability is zero.

* At the other extreme, if treatment is perfectly effective, the probability is 100%.

It might be frustrating that we can't be more specific about the probability of the counterfactual, but if someone you know was diagnosed late and died, and it bothers you to think they would have lived if they had been diagnosed earlier, it might be some comfort to realize that we don't know that -- and it could be unlikely.


## Comparing Survival Rates

Now let's consider the last incorrect inference.

> If 5-year survival increases over time, it is tempting to conclude that treatment has improved.

This conclusion is appealing because if cancer treatment improves, survival rates improve, other things being equal.
But if we do more screening and catch more cancers early, survival rates improve even if treatment is no more effective.
And if screening becomes more sensitive, and detects smaller tumors, survival rates also improve.

For many cancers, all three factors have changed over time: improved treatment, more screening, and more sensitive screening.
Looking only at survival rates, we can't tell how much change in survival we should attribute to each.


And the answer is different for different sites.
For example, [this paper](https://pubmed.ncbi.nlm.nih.gov/25417232/) concludes:

>  In some cases, increased survival was accompanied by decreased burden of disease, reflecting true progress. For example, from 1975 to 2010, five-year survival for colon cancer patients improved ... while cancer burden fell: Fewer cases ... and fewer deaths ..., a pattern explained by both increased early detection (with removal of cancer precursors) and more effective treatment. In other cases, however, increased survival did not reflect true progress. In melanoma, kidney, and thyroid cancer, five-year survival increased but incidence increased with no change in mortality. This pattern suggests overdiagnosis from increased early detection, an increase in cancer burden. 

And [this paper](https://pubmed.ncbi.nlm.nih.gov/10865276/) explains:

> Screening detects abnormalities that meet the pathological definition of cancer but that will never progress to cause symptoms or death (non-progressive or slow growing cancers). The higher the number of overdiagnosed patients, the higher the survival rate.

It concludes that "Although 5-year survival is a valid measure for comparing cancer therapies in a randomized trial, our analysis shows that changes in 5-year survival over time bear little relationship to changes in cancer mortality. Instead, they appear primarily related to changing patterns of diagnosis."

That conclusion might be stated too strongly. 
[This response paper](https://www.researchgate.net/publication/46555895_Are_Increasing_5-Year_Survival_Rates_Evidence_of_Success_Against_Cancer_A_Reexamination_Using_Data_from_the_US_and_Australia) concludes "While the change in the 5-year survival rate is not a perfect measure of progress against cancer [...] it does contain useful information; its critics may have been unduly harsh. Part of the long-run increase in 5-year cancer survival rates is due to improved [...] therapy."

But they acknowledge that with survival rates alone, we can't say what part -- and in some cases we have evidence that it is small.


## Summary

Survival statistics are misleading because they suggest inferences they do not actually support.

* Survival rates from the past might not apply to the present, and for a particular patient, the probability of survival depends on (1) causal factors like general health, and (2) informational factors like the mode of discovery (screening vs symptomatic presentation).

* If survival rates are higher when tumors are discovered early, that doesn't mean that more screening would be better.

* And if a cancer is diagnosed late, and the patient dies, that doesn't mean that if it had been diagnosed early, they would have lived.

* Finally, if survival rates improve over time (or they are different in different places) that doesn't mean treatment is more effective.

To be clear, all of these conclusions *can* be true, and in some cases we know they are true, at least in part.
For some cancers, treatments have improved, and for some, additional screening would save lives.
But to support these conclusions, we need other methods and metrics -- notably randomized controlled trials that compare mortality.
Survival rates alone provide little or no information, and they are more likely to mislead than inform.


## Markov analysis with PyMC

Instead of QuantEcon, we could have used PyMC to simulate the Markov chain.

```python
# pip install pymc pymc-extras

import pymc as pm
import pymc_extras as pmx

# map labels <-> ints
state_to_idx = {s: i for i, s in enumerate(states)}
idx_to_state = np.array(states)

# initial distribution: start in 'U1'
init_probs = np.zeros(len(states))
init_probs[state_to_idx['U1']] = 1.0
```

```python
T = 100    # number of time steps     
N = 10     # number of trajectories

with pm.Model():
    init_dist = pm.Categorical.dist(p=init_probs)
    chain_rv = pmx.DiscreteMarkovChain("chain", P=P, init_dist=init_dist, shape=(T,))

# Fast, no idata:
traj = pm.draw(chain_rv, draws=N)
```

```python
traj_labels = idx_to_state[traj]
traj_labels[0]
```

<!-- #region tags=["remove-print"] -->
Copyright 2025 Allen B. Downey

License: [Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
<!-- #endregion -->

```python

```
