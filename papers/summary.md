# Summary: Capture-Recapture for Human Populations (Chao, 2015)

**Paper:** Anne Chao, *Capture-Recapture for Human Populations*, Wiley StatsRef (2014–2015 update).  
**Full text:** [`chao2015.md`](chao2015.md) (converted from [`chao2015.pdf`](chao2015.pdf)).

---

## What the paper is about

Capture–recapture methods from ecology are applied to **human populations** when several incomplete lists of cases (hospitals, labs, questionnaires, registries, etc.) are available. Merging lists and dropping duplicates undercounts the true size `N`. The paper reviews history and assumptions, then focuses on Chao’s **sample-coverage** approach for estimating `N` and measuring dependence among lists, implemented in the R package **CARE1**. Two examples: a **hepatitis A** outbreak (low overlap) and **diabetes** in Italy (high overlap).

---

## Setup and notation

- Unknown closed population size `N`.
- `t` incomplete lists; each person is present (`1`) or absent (`0`) on each list.
- Cell counts `Z_{s1…st}` for each capture pattern; the all-zero cell `Z_{00…0}` is unobserved (the undercount).
- `M` = number of distinct people seen on at least one list; goal is to estimate `Z_{00…0}` or equivalently `N = M + Z_{00…0}`.
- Assumptions: independence of individuals; consistent case definition; correct matching; roughly constant `N` during the study; every person has positive probability of appearing on each list.

**Dependence** between lists (which biases naive estimators) comes from:

1. **Local dependence** — being on one list causally changes the chance of being on another (e.g., a positive serum test sends someone to hospital).
2. **Heterogeneity** — people differ in ascertainment probabilities; aggregating independent strata can induce dependence (Simpson-like).

These two sources are usually confounded. Positive dependence → underestimation of `N` under independence; negative dependence → overestimation. With only two lists, dependence is not identifiable without extra assumptions; **at least three lists** are needed to model it.

---

## Main methods (brief)

| Approach | Role |
|----------|------|
| **Two-list Petersen / Chapman** | Classic `N ≈ n1 n2 / m2` (Chapman bias-corrected). Useful pairwise diagnostics when `t > 2`. |
| **Sample coverage (focus of paper)** | Quantifies overlapping fraction `C`; estimators `N̂0` (independence), `N̂` (high coverage), `N̂1` (one-step / often a bound when coverage is low). |
| **Ecological models (Mt, Mb, Mh, …)** | Wildlife-style capture-probability models; briefly reviewed. |
| **Log-linear models** | Contingency-table models for multi-list dependence; briefly reviewed. |

Rule of thumb for coverage estimators: if estimated coverage `Ĉ ≳ 55%` and bootstrap SE of `N̂` is not huge, use `N̂`; otherwise data are thin and `N̂1` is recommended (often a **lower bound** when dependence is positive, as in most epi applications).

---

## Hepatitis A example (main focus)

### Setting

Outbreak of hepatitis A in and around a college in **northern Taiwan, April–July 1995**. Analysis restricted to **students** of that college. Three incomplete case lists:

| List | Source | Cases `nj` |
|------|--------|------------|
| **P** | Serum test, Institute of Preventive Medicine of Taiwan | 135 |
| **Q** | National Quarantine Service (local hospital reports) | 122 |
| **E** | Epidemiologist questionnaires | 126 |

**Distinct ascertained cases:** `M = 271`.

### Crosstab (Table 1)

| P | Q | E | Count |
|---|---|---|-------|
| 1 | 1 | 1 | `Z111 = 28` |
| 1 | 1 | 0 | `Z110 = 21` |
| 1 | 0 | 1 | `Z101 = 17` |
| 1 | 0 | 0 | `Z100 = 69` |
| 0 | 1 | 1 | `Z011 = 18` |
| 0 | 1 | 0 | `Z010 = 55` |
| 0 | 0 | 1 | `Z001 = 63` |
| 0 | 0 | 0 | `Z000 = ??` |

Same counts as in *Think Bayes* Chapter 15 (`data3 = [0, 63, 55, 18, 69, 17, 21, 28]` in reverse category order).

**Why dependence is plausible here:** a student with a positive serum test (P) is more likely to seek hospital care and appear on Q — **local positive dependence** between P and Q.

### CARE1 results (Section 6.1)

Pairwise Petersen/Chapman estimates are all in a **narrow band (~331–378)**, so pairwise comparisons do not clearly flag the direction of dependence.

Sample-coverage output:

| Estimator | Meaning | Estimate | Approx. 95% CI | Notes |
|-----------|---------|----------|----------------|-------|
| `Ĉ` | Overlapping fraction | **0.513** | — | Low (<55%) |
| `D` | Average overlapping cases | 208.667 | — | |
| **N̂0** | Independence | **407** | (365, 467) | SE ≈ 26 |
| **N̂** | High-coverage formula | **971** | (411, 3778) | SE ≈ 688 — **unstable** |
| **N̂1** | One-step (low coverage) | **508** | (425, 636) | SE ≈ 53; **recommended lower bound** |

Mean ascertainment probabilities for the three lists are similar (~0.25–0.33 depending on which `N` is plugged in). Pairwise CCV (dependence) estimates are **positive**, so independence-based `N̂0` is expected to **underestimate**.

### Ground truth

After the three surveys, Taiwan’s National Quarantine Service ran a **campus-wide serum screen**. The conclusive infected count was about **545**. Chao’s recommended lower bound `N̂1 = 508` is close; the example shows both the need for undercount correction and that low-overlap data may only support a bound, not a precise point estimate.

---

## Contrast: diabetes example (high overlap)

Four lists in an Italian community (`M = 2069`). Estimated coverage **~80%**. Recommended `N̂ = 2609` (95% CI about 2477–2784). Strong positive dependence for some list pairs (especially involving purchases of strips/syringes). Shows that with sufficient overlap, coverage-based adjustment can give a usable point estimate — unlike the HAV case.

---

## Takeaways for the Think Bayes / PyMC hepatitis notebook

1. **Data and story** match Chapter 15: three lists, `M = 271`, unknown `Z000`, estimate total infections `N`.
2. The chapter’s **shared-`p` multinomial / independence-style model** is closest in spirit to estimators that ignore list dependence (e.g. `N̂0 ≈ 407`). Expect something in that ballpark, not Chao’s dependence-adjusted `N̂1 = 508` or the true ~545, unless the model allows dependence or heterogeneous ascertainment.
3. **Known truth ≈ 545** is unusually valuable for checking Bayesian posteriors (mean / HDI vs Chao and vs truth).
4. Low overlap and positive dependence are substantive features of the example, not just numerical details — worth mentioning in prose even if the first PyMC model is the simple equal-`p` version from the book.
5. Natural extensions after the book model: list-specific `p`s (Lincoln-style); models that allow dependence / heterogeneity and compare to Chao’s `N̂0` / `N̂1` / 545.

---

## Citation

Chao, A. (2015). Capture-Recapture for Human Populations. *Wiley StatsRef: Statistics Reference Online*. https://doi.org/10.1002/9781118445112.stat04855.pub2

Underlying HAV analysis also discussed in Chao et al. (2001), *Statistics in Medicine* 20:3123–3157 (ref. [22] in the paper).
