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

```python
import os
import pandas as pd
import numpy as np
import glob
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from termcolor import colored
```

# Attempt to distinguish front/side light

NOTE: this approach is necessary to do manually.
Every first is front light, every second is side light - we have to split them

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
```

```python
for root, dirs, files in os.walk("/Users/gimli/cvr/data/beton/CM1-cracks/2026_06_28-204117"):
    for file in files:
        if file.endswith("samples.csv") and "6B2_rear" not in root and "6C1_front" not in root:
            key = "_".join(file.split("_")[0:2])
            raw_measurements[key] = pd.read_csv(os.path.join(root, file))
```

```python
# Annotate with front/side light
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
ref_data = data[data["light_direction"] == "side_light"]
```

```python
ref_data
```

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
ref_data[ref_data["group_name"] == "1C1_front"]
```

```python
plt.figure(figsize=(15, 5))
plt.plot(xgrid, uncertainty_curve["repeatability_limit"], label="repeatability limit")
plt.plot(xgrid, uncertainty_curve["single_measurement_95"], label="single measurement 95%")
plt.xlabel("Mean value for measured sample (skeleton length px)")
plt.ylabel("95% range of repeated measurements (±px)")
plt.legend()
plt.title("Bubble area (px) measurement stability", fontsize=20)
plt.show()
```

```python
test=[]
test.append(pd.read_csv("/Users/gimli/cvr/data/beton/CM1-cracks/overeni/2026_06_27-121406/6B2_rear/6B2_rear_samples.csv"))
test[0].drop(index=test[0].index[0], axis=0, inplace=True)
test.append(pd.read_csv("/Users/gimli/cvr/data/beton/CM1-cracks/overeni/2026_06_27-121406/6C1_front/6C1_front_samples.csv"))
```

```python
#pd.concat(test).to_excel("/Volumes/FUEL_BETON_TEAM/CM1/overeni/report.xlsx")
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
            print(f"({en}/{ev}){value_col}: {v} -> ({interval[0]:.2f}-{interval[1]:.2f})")

```

```python
for key, values in matches.items():
    print(f"{key}: {np.sum(values)} of fails...")
```

```python
gx03 = np.load("/Users/gimli/cvr/data/beton/CM1-sample/GX012203_part0.mp4.npy")
```

```python
plt.figure(figsize=(15, 15))
#plt.imshow(gx91[100], cmap="gray")
plt.imshow(np.min(gx03, axis=0), cmap="gray")
plt.xlim(500, 1650)
plt.show()
```

```python
plt.figure(figsize=(15, 15))
#plt.imshow(gx91[100], cmap="gray")
plt.imshow(gx03[-7], cmap="gray")
plt.xlim(500, 1650)
plt.show()
```

```python
len(gx03)
```

```python

```
