---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.17.3
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Crack feature variability

**Purpose of this notebook:** Testing of a new technology.

**Background:** Tested technology is able to measure crack-features on a concrete sample. Samples are scanned by GoPro kamera in 4K resolution. Video frames are registered into a matrix where pixels belonging to the same physical surface point are in the same row of registered matrix. Matrix is then reduced into image suitable for cracks and bubbles segmentation. On a segmentation mask these features are described by - crackSkeletonSum, crackBoundaryLength, bubbleAreaSum, bubbleBoundary and bubbleCount.

**Requirements:** Training and testing data in a form of csv files. Each file contains 7 columns: sample_id, skeletonSum_px, crackBoundaryLength_px, bubbleAreaSum_px, bubbleBoundaryLength_px, bubbleCount, group_name.

**Content:** Training data creates the intervals of confidence against which the testing samples are compared.

```python
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```

## Filtration of training data according to the light direction
(direct/side light)

According to acquisition documentation, every first is direct light, every second is side light
In case of unavailable raw data directory, this can be skipped. If skipped the computed intervals will be slightly bigger (there is a difference between crack visibility with direct and side light).

```python
full_paths = []
for root, dirs, files in os.walk("/Volumes/FUEL_BETON_TEAM/CM1/scan"):
    for file in files:
        if file.endswith("MP4"):
            full_paths.append(os.path.join(root, file))

records = {}
for fp in full_paths:
    parts = fp.split(os.path.sep)
    sample_id = parts[-3]
    if sample_id not in records:
        records[sample_id] = {}

    side = parts[-2]
    if side not in records[sample_id]:
        records[sample_id][side] = []

    filename = parts[-1]
    records[sample_id][side].append(filename)

for rid in records.keys():
    for sid in records[rid].keys():
        records[rid][sid] = sorted(records[rid][sid])

front_light = [video for rid in records.keys() for sid in records[rid].keys() for video in records[rid][sid][::2]]
side_light = [video for rid in records.keys() for sid in records[rid].keys() for video in records[rid][sid][1::2]]

```

# Crack data


## Data load

```python
raw_measurements = {}
# NAS data storage
data_storage = "/Volumes/FUEL_BETON_TEAM/CM1/measurements/2026_06_28-204117"
# Local data copy
data_storage = "/Users/gimli/cvr/data/beton/CM1-cracks/2026_06_28-204117"
```

```python
for root, dirs, files in os.walk(data_storage):
    for file in files:
        if file.endswith("samples.csv") and "6B2_rear" not in root and "6C1_front" not in root:
            key = "_".join(file.split("_")[0:2])
            raw_measurements[key] = pd.read_csv(os.path.join(root, file))
```

```python
# Annotate with front/side light if available
for key in raw_measurements.keys():
    light_directions = []
    for sid in raw_measurements[key]["sample_id"]:
        original_filename = f"{sid.split("-")[1].split("_")[0]}.MP4"
        if original_filename in side_light:
            light_directions.append("side_light")
        elif original_filename in front_light:
            light_directions.append("front_light")
        else:
            light_directions.append("unknown")
    raw_measurements[key]["light_direction"] = light_directions
```

```python
data = pd.concat([raw_measurements[key] for key in raw_measurements])
```

```python
data
```

```python
## data filtration if available
preferred_light_direction = "side_light" if side_light else "unknown"
ref_data = data[data["light_direction"] == preferred_light_direction]
```

```python
ref_data
```

## Creation of confidence intervals

## Repeatability model for repeated scan-derived measurements

Let $y_{ij}$ denote one measured scalar feature from video/scan $j$ of physical sample $i$, for example total crack skeleton length, crack boundary length, bubble area, bubble boundary length, or bubble count. Each physical sample has $n_i$ repeated measurements, typically $n_i \in {2,\dots,5}$. The measurement model is

$$y_{ij} = \mu_i + \varepsilon_{ij},$$

where $\mu_i$ is the latent sample-specific value and $\varepsilon_{ij}$ is the acquisition and post-processing error. The aim is not to estimate the uncertainty of the mean of one sample, but the repeatability uncertainty of a new measurement acquired and processed using the same pipeline.

For each sample, first estimate the sample mean

$$\bar{y}_i = \frac{1}{n_i}\sum_{j=1}^{n_i} y_{ij}.$$

The within-sample residuals are

$$r_{ij} = y_{ij} - \bar{y}_i.$$

These residuals remove the dominant between-sample variation and isolate the variability due to repeated acquisition and processing. For sample $i$, calculate the within-sample sum of squares

$$SS_i = \sum_{j=1}^{n_i} (y_{ij}-\bar{y}_i)^2$$

and the corresponding degrees of freedom

$$\nu_i = n_i - 1.$$

The sample-specific repeatability variance estimate is

$$s_i^2 = \frac{SS_i}{\nu_i}.$$

Because $n_i$ is small, $s_i^2$ is noisy and should not be interpreted independently for individual samples.

## Local heteroscedastic repeatability estimation

The repeatability error may depend on the magnitude of the measured quantity. Instead of assuming a constant variance, we estimate a magnitude-dependent repeatability variance

$$\operatorname{Var}(\varepsilon_{ij}\mid \mu_i) = \sigma^2(\mu_i).$$

Since $\mu_i$ is unknown, the observed sample mean $\bar{y}_i$ is used as its proxy. For any measurement level $x$, select a local neighborhood $N_K(x)$ containing the $K$ samples whose means $\bar{y}_i$ are closest to $x$. The local pooled variance is then estimated from within-sample sums of squares:

$$\hat{\sigma}^2(x) =
\frac{
\sum_{i \in N_K(x)} w_i(x) SS_i
}{
\sum_{i \in N_K(x)} w_i(x) \nu_i
}.
$$

Here $w_i(x)$ is an optional distance-dependent weight. With unweighted KNN,

$$w_i(x)=1.$$

With triangular distance weighting inside the selected KNN neighborhood,

$$w_i(x) =
\max\left(
\epsilon,
1-\frac{|\bar{y}_i-x|}{h(x)}
\right),
$$

where

$$h(x)=\max_{i \in N_K(x)} |\bar{y}_i-x|$$

is the local KNN bandwidth and $\epsilon$ is a small positive constant preventing the outermost neighbor from receiving exactly zero weight. The corresponding local repeatability standard deviation is

$$\hat{\sigma}(x)=\sqrt{\hat{\sigma}^2(x)}.$$

This estimator pools only within-sample variability. Raw measurements from neighboring samples must not be pooled directly, because differences between physical samples would otherwise inflate the repeatability estimate.

## Repeatability intervals

For a single new measurement $y$, the local repeatability standard deviation is evaluated at approximately $x=y$, or at the local mean if multiple repeats of the new sample are available. A 95% single-measurement repeatability uncertainty interval around the latent sample value is approximated by

$$y \pm 1.96 \hat{\sigma}(y).$$

This interval describes the expected deviation of one acquisition and processing result from the latent sample value, assuming approximately symmetric residuals.

The repeatability limit for the difference between two independent measurements of the same sample is

$$r(y) = 1.96 \sqrt{2} \hat{\sigma}(y) \approx 2.77\hat{\sigma}(y). $$

Thus, two independently acquired and processed scans of the same physical sample are expected to differ by less than $r(y)$ in approximately 95% of repeated pairs.

For an average of $m$ repeated measurements of a new sample,

$$\bar{y}_{\text{new}} = \frac{1}{m}\sum_{j=1}^{m}y_j,$$

the repeatability standard error of the mean is

$$SE_{\bar{y}} = \frac{\hat{\sigma}(\bar{y}_{\text{new}})}{\sqrt{m}},$$

and the approximate 95% interval for the latent value is

$$\bar{y}_{\text{new}} \pm 1.96\frac{\hat{\sigma}(\bar{y}_{\text{new}})}{\sqrt{m}}.$$


```python
def prepare_repeatability_groups(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
) -> pd.DataFrame:
    """Calculate group means and within-group repeatability information."""
    records = []

    for group_name, values in df.groupby(group_col)[value_col]:
        values = values.dropna().to_numpy(dtype=float)
        n = values.size

        if n < 2:
            continue

        mean = values.mean()
        ss = np.sum((values - mean) ** 2)

        records.append({
            group_col: group_name,
            "mean": mean,
            "n": n,
            "df": n - 1,
            "ss_within": ss,
            "variance": ss / (n - 1),
        })

    return pd.DataFrame(records)

def knn_repeatability_sd(
    group_stats: pd.DataFrame,
    x: float,
    k: int = 15,
    weighted: bool = False,
) -> dict:
    """
    Estimate repeatability SD locally around measurement level x.

    Parameters
    ----------
    group_stats:
        Output of prepare_repeatability_groups().
    x:
        Measurement level at which repeatability is estimated.
    k:
        Number of neighboring physical samples.
    weighted:
        If True, use distance-dependent triangular weights.
    """
    if len(group_stats) == 0:
        raise ValueError("No groups with at least two measurements.")

    data = group_stats.copy()
    data["distance"] = np.abs(data["mean"] - x)

    neighbors = (
        data.nsmallest(min(k, len(data)), "distance")
        .copy()
    )

    if not weighted or len(neighbors) == 1:
        neighbors["weight"] = 1.0
    else:
        bandwidth = neighbors["distance"].max()

        if bandwidth == 0:
            neighbors["weight"] = 1.0
        else:
            # Triangular kernel inside the KNN-selected neighborhood.
            neighbors["weight"] = (
                1.0 - neighbors["distance"] / bandwidth
            ).clip(lower=0)

            # Avoid a zero contribution from the outermost neighbor.
            neighbors["weight"] = np.maximum(
                neighbors["weight"],
                1e-8,
            )

    numerator = np.sum(
        neighbors["weight"] * neighbors["ss_within"]
    )
    denominator = np.sum(
        neighbors["weight"] * neighbors["df"]
    )

    variance = numerator / denominator

    return {
        "x": x,
        "repeatability_sd": np.sqrt(variance),
        "variance": variance,
        "effective_df": denominator,
        "n_groups": len(neighbors),
        "mean_min": neighbors["mean"].min(),
        "mean_max": neighbors["mean"].max(),
    }
```

## Choice of K and diagnostics

The neighborhood size $K$ controls the bias-variance trade-off of the local repeatability curve. Small $K$ follows local changes more closely but produces noisy uncertainty estimates. Large $K$ gives a smoother curve but may oversmooth real heteroscedasticity.

A recommended diagnostic is group-wise leave-one-sample-out validation. For each sample $i$, estimate $\hat{\sigma}_{-i}(\bar{y}_i)$ using all other samples, then calculate standardized residuals

$$z_{ij} = \frac{y_{ij}-\bar{y}_i}{\hat{\sigma}_{-i}(\bar{y}_i)}.$$

If the local variance model is adequate, the standardized residuals should have approximately stable spread over the measurement range. Systematic widening or narrowing of $z_{ij}$ as a function of $\bar{y}_i$ indicates under- or overestimation of local repeatability.

Candidate values of $K$ can be compared by minimizing the leave-one-sample-out negative log-likelihood-like score

$$Q(K) = \sum_i\left[\nu_i \log \hat{\sigma}_{-i}^2(\bar{y}_i) + \frac{SS_i}{\hat{\sigma}_{-i}^2(\bar{y}_i)} \right].$$

Lower $Q(K)$ indicates better predictive calibration of the repeatability variance. Because each sample contributes only $n_i-1$ variance degrees of freedom, the selected local neighborhood should usually include enough samples to provide at least approximately 20–30 total within-sample degrees of freedom.

```python
def select_knn_k(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    k_values=range(5, 31),
) -> pd.DataFrame:
    group_stats = prepare_repeatability_groups(
        df,
        group_col,
        value_col,
    )

    results = []

    for k in k_values:
        score = 0.0
        used_groups = 0

        for _, held_out in group_stats.iterrows():
            training = group_stats.loc[
                group_stats[group_col] != held_out[group_col]
            ]

            if len(training) == 0:
                continue

            local = knn_repeatability_sd(
                training,
                x=held_out["mean"],
                k=min(k, len(training)),
                weighted=True,
            )

            variance = max(local["variance"], 1e-12)

            score += (
                held_out["df"] * np.log(variance)
                + held_out["ss_within"] / variance
            )
            used_groups += 1

        results.append({
            "k": k,
            "cv_score": score,
            "used_groups": used_groups,
        })

    return pd.DataFrame(results).sort_values("cv_score")
```

```python
ref_data
```

```python
select_knn_k(ref_data, data.columns[6], data.columns[1])["k"].to_numpy()[0]
```

<!-- #region -->
# Single-sample repeatability check

This section describes a quality-control test for repeated measurements of one physical sample. The goal is to determine whether the variability observed across repeated acquisitions and post-processing runs of the same sample is compatible with the expected repeatability of the measurement pipeline.

Let

$$
y_1, y_2, \dots, y_n
$$

denote repeated measurements of the same scalar feature from one physical sample, for example total crack skeleton length, crack boundary length, bubble area, bubble boundary length, or bubble count. The measurements are assumed to follow

$$
y_j = \mu + \varepsilon_j,
$$

where $\mu$ is the latent value of the physical sample and $\varepsilon_j$ is the acquisition and post-processing error.

The expected repeatability standard deviation is obtained from the previously estimated global or local repeatability model. If the repeatability is magnitude-dependent, the expected standard deviation is evaluated at the mean level of the tested sample:

$$
\hat{\sigma}_{\mathrm{rep}} =
\hat{\sigma}(\bar{y}).
$$

The repeated measurements of the tested sample are summarized by their mean

$$\bar{y} = \frac{1}{n} \sum_{j=1}^{n} y_j$$

and within-sample residuals

$$r_j = y_j - \bar{y}.$$

The observed within-sample sum of squares is

$$SS = \sum_{j=1}^{n}\left(y_j-\bar{y}\right)^2.$$

The observed repeatability variance and standard deviation are

$$s^2=\frac{SS}{n-1},$$

$$s=\sqrt{s^2}.$$

The main diagnostic quantity is the ratio between observed and expected repeatability standard deviation:

$$\rho=\frac{s}{\hat{\sigma}_{\mathrm{rep}}}.$$

Interpretation:

* $\rho \approx 1$: observed repeatability is consistent with the expected pipeline repeatability;
* $\rho < 1$: repeated measurements are more stable than expected;
* $\rho > 1$: repeated measurements are more variable than expected;
* large values of $\rho$ indicate possible acquisition instability, processing failure, sample movement, illumination inconsistency, segmentation instability, or another sample-specific artefact.

A formal one-sided excess-variability test can be based on the statistic

$$\chi^2=\frac{SS}{\hat{\sigma}_{\mathrm{rep}}^2}.$$

Under the assumption that the tested sample follows the expected repeatability model and that residuals are approximately Gaussian,

$$\chi^2 \sim \chi^2_{n-1}.$$

The upper-tail p-value is

$$p_{\mathrm{excess}} = P\left(\chi^2_{n-1}\geq\frac{SS}{\hat{\sigma}_{\mathrm{rep}}^2}\right).$$

A small value of $p_{\mathrm{excess}}$ indicates that the observed scatter of repeated measurements is larger than expected from the repeatability model.

The test is intentionally one-sided, because unusually low variability is usually not a failure mode. The relevant quality-control question is whether the repeated measurements are more variable than expected.


## Notes

For small numbers of repeats, especially $n=2$ to $5$, the chi-square test has low resolution. The ratio

$$
\rho =
\frac{s}{\hat{\sigma}_{\mathrm{rep}}}
$$

is therefore often more informative than the p-value.

The repeatability model $\hat{\sigma}(x)$ should be estimated from independent historical or calibration samples. If the tested sample was used to fit the repeatability model, it should be excluded from the local model estimation when performing an unbiased quality-control check.

For magnitude-dependent repeatability, $\hat{\sigma}(\bar{y})$ may be obtained using a local KNN variance model, spline model, regression model, or another heteroscedastic variance estimator. The single-sample check itself is independent of how the expected repeatability curve was estimated.

<!-- #endregion -->

```python
fields = ['sample_id', 'skeletonSum_px', 'crackBoundaryLengthSum_px',
       'bubbleAreaSum_px', 'bubbleBoundaryLengthSum_px', 'bubbleCount',
       'group_name', 'light_direction']
ref_data = data[data["light_direction"] == "front_light"]
for field in fields[1:-2]:
    group = "group_name"
    k = select_knn_k(ref_data, group, field, range(5, 40))["k"].to_numpy()[0]
    print(f"KNN k: {k}")

    rgs = prepare_repeatability_groups(ref_data, group, field)
    xgrid = np.linspace(rgs["mean"].min(), rgs["mean"].max(), 30)
    uncertainty_curve = pd.DataFrame([knn_repeatability_sd(rgs, x, k, True)
                                      for x in xgrid])
    uncertainty_curve["single_measurement_95"] = (
        1.96 * uncertainty_curve["repeatability_sd"]
    )

    uncertainty_curve["repeatability_limit"] = (
        1.96 * np.sqrt(2)
        * uncertainty_curve["repeatability_sd"]
    )

    plt.figure(figsize=(15, 5))
    plt.plot(xgrid, uncertainty_curve["repeatability_limit"], label="repeatability limit")
    plt.plot(xgrid, uncertainty_curve["single_measurement_95"], label="single measurement 95%")
    plt.xlabel(f"Mean value for measured sample ({field})")
    plt.ylabel("95% range of repeated measurements (±px)")
    plt.legend()
    plt.title(f"{field} measurement stability", fontsize=20)
    plt.show()
```

```python
# NAS test records
test_records = "/Volumes/FUEL_BETON_TEAM/CM1/measurements/2026_06_27-121406"
# Local test records
test_records = "/Users/gimli/cvr/data/beton/CM1-cracks/overeni/2026_06_27-121406"
```

```python
test=[]
test.append(pd.read_csv(os.path.join(test_records, "6B2_rear", "6B2_rear_samples.csv")))
test[0].drop(index=test[0].index[3], axis=0, inplace=True)
test.append(pd.read_csv(os.path.join(test_records, "6C1_front", "6C1_front_samples.csv")))
```

```python
pd.concat(test)
```

```python
def measurement_interval(
    measured_value: float,
    group_stats: pd.DataFrame,
    k: int = 15,
) -> tuple[float, float]:
    result = knn_repeatability_sd(
        group_stats,
        x=measured_value,
        k=k,
        weighted=True,
    )

    margin = 1.96 * result["repeatability_sd"]

    return (
        measured_value - margin,
        measured_value + margin,
    )
```

```python
matches = {}
for value_col in data.columns[1:-2]:
    matches[value_col] = []
    group_stats = prepare_repeatability_groups(
        data,
        "group_name",
        value_col,
    )
    k = select_knn_k(data, "group_name", value_col, range(5, 40))

    for en, t in enumerate(test):
        interval = measurement_interval(t[value_col].mean(), group_stats, k["k"].to_numpy()[0])
        for ev, v in enumerate(t[value_col].to_numpy()):
            matches[value_col].append(bool(v < interval[0] or v > interval[1]))
            #if bool(v < interval[0] or v > interval[1]):
            print(f"({en}/{ev}){value_col}: {v} -> ({interval[0]:.2f} - {interval[1]:.2f})")

```

```python
for key, values in matches.items():
    print(f"{key}: {np.sum(values)} of fails...")
```

## Optional transformation

If the absolute repeatability SD increases approximately proportionally to the measured value, the analysis can be performed on a transformed scale. For strictly positive values,

$$z_{ij} = \log(y_{ij})$$

models multiplicative measurement error. For values that may include zero,

$$z_{ij} = \log(1+y_{ij})$$

is usually more stable.

The repeatability model is then fitted to $z_{ij}$. Intervals can be transformed back to the original scale. For a log-scale repeatability SD $\hat{\sigma}_{\log}$, the approximate multiplicative 95% single-measurement interval is

$$y \cdot \left[\exp(-1.96\hat{\sigma}*{\log}), \exp(+1.96\hat{\sigma}*{\log}) \right].$$

For count variables such as `bubbleCount`, the same empirical approach can be used as a pragmatic approximation, but a Poisson or negative-binomial mean-variance model may be preferable if the counts are low or strongly discrete.

```python

```
