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
%load_ext autoreload
%autoreload 2
```

```python
import numpy as np
import cv2
from tqdm.auto import tqdm
from checkerboard_corners.preprocessor import otsu_based_thresholding, light_balance
from checkerboard_corners.detector import find_corners, detect_checkerboard, local_maxima
from checkerboard_corners.registration.warping import from_checkerboard_corners_to_image_coordinates, SmoothTPSWarp
from checkerboard_corners.topology_gpt import reconstruct_checkerboard_topology, draw_reconstruction
from checkerboard_corners.registration.warping import SmoothTPSWarp
import matplotlib.pyplot as plt
import os
import matplotlib.colors as cls
```

```python
args = {
    "checkerboard_detection": {
      "harris_k": 0.02,
      "subpix_window": 9,
      "corner_detection_safety_margin_tblr_px": np.array([40, 40, 1000, 500]).astype(int), # top, bottom, left, right
      "square_size": 280.0,
      "cut_radius": 25,
      "ring_radius": 35,
      "dedup_radius": 0.5,
      "enable_missing_corner_completion": True,
  },
  "thresholding": {
      "otsu_factor": .4,
      "kernel": (2, 2),
  },
}
```

```python
input_video_6B2 = "/Users/gimli/cvr/data/beton/CM1-sample/upravená/GX012208.MP4"
input_video_6C1 = "/Users/gimli/cvr/data/beton/CM1-sample/upravená/GX012209.MP4"
```

```python
def detect_checkerboard_corners(input_video, args):
    # Read some frame
    scale = 12
    vidcap = cv2.VideoCapture(input_video)
    vidcap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    success, frame = vidcap.read()
    data = []
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    tblr = args["checkerboard_detection"]["corner_detection_safety_margin_tblr_px"]
    for fid in tqdm(range(frame_count), total=frame_count, desc="Checkerboard corners"):
        if not success:
            break
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cropped_image = image[tblr[0]:-tblr[1], tblr[2]: -tblr[3]]
        resized = cv2.resize(cropped_image, (cropped_image.shape[1] // scale, cropped_image.shape[0] // scale), interpolation=cv2.INTER_LANCZOS4)
        mask = np.ones_like(resized)
        corner_mask = np.zeros_like(resized)

        thr, t = otsu_based_thresholding(resized, mask,
                                         otsu_factor=1, #args["thresholding"]["otsu_factor"],
                                         kernel=args["thresholding"]["kernel"])

        corners_subpix = find_corners(
                resized, thr, mask,
                subpix_window=args["checkerboard_detection"]["subpix_window"],
                harris_k=args["checkerboard_detection"]["harris_k"],
                local_maxima_count=150
            )

        if len(corners_subpix) == 0:
            data.append((image, np.array([]), np.array([]), None))
            print(f"{fid}: No corners found.")
        else:
            corners_subpix = cv2.cornerSubPix(
                cv2.GaussianBlur(cropped_image, (101, 101), 11),
                np.float32((np.array(corners_subpix) * scale).reshape(-1, 1, 2).tolist()),
                (args["checkerboard_detection"]["subpix_window"] * 3,
                 args["checkerboard_detection"]["subpix_window"] * 3),
                (1, 1),
                (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.2)
            )
            try:
                image_points, ref_points, debug = reconstruct_checkerboard_topology(
                    gray=cropped_image,
                    binary=cropped_image > t,
                    corners=corners_subpix.reshape(-1, 2),
                    square_size=args["checkerboard_detection"]["square_size"],
                    cut_radius=args["checkerboard_detection"]["cut_radius"],
                    ring_radius=args["checkerboard_detection"]["ring_radius"],
                    dedup_radius=args["checkerboard_detection"]["dedup_radius"],
                    enable_missing_corner_completion=args["checkerboard_detection"]["enable_missing_corner_completion"]
                )
                data.append((image, image_points, ref_points, debug))
                print(f"{fid}: {len(ref_points)} corners.")
            except Exception as e:
                print(f"{fid}: {e}")
                data.append((image, np.array([]), np.array([]), None))
                print(f"{fid}: No corners recorded.")
        success, frame = vidcap.read()
    vidcap.release()

    return data
```

```python
data6B2 = detect_checkerboard_corners(input_video_6B2, args)
```

```python
data6C1 = detect_checkerboard_corners(input_video_6C1, args)
```

```python
plt.figure(figsize=(15, 5))
plt.imshow(data6C1[150][0], cmap="gray")
plt.scatter(data6C1[150][1][:, 0] + tblr[2],
            data6C1[150][1][:, 1] + tblr[0], c="r", s=30, marker="+")
plt.show()
```

```python
def genetic_calibration_subset(
        img_points, ref_points,
        subset_size: int,
        population_size: int,
        mutation: int,
        replace_ratio: float,
        generations: int = 200,
        img_size=None,
        rng_seed: int = 42,
        elitism: int = 2,
        tournament_k: int = 3,
):
    """
    Genetic search of a frame subset that minimizes calibration reprojection RMS.

    Parameters
    ----------
    groups : list[tuple[np.ndarray, np.ndarray]]
        Per-frame tuples as (objpoints_float32, imgpoints_float32).
        Each objpoints: (n_pts, 3), dtype float32.
        Each imgpoints: (n_pts, 2), dtype float32.
    subset_size : int
        Number of frames (indices into `groups`) per individual.
    population_size : int
        Number of individuals in the genetic population.
    mutation : int
        Number of frame indices to replace randomly during mutation (>=1).
    replace_ratio : float
        Fraction of the population replaced each generation (0..1].
        New individuals are formed via selection + crossover + mutation.
    generations : int, optional
        Number of GA generations (default 200).
    img_size : tuple[int,int] | None, optional
        (width, height). If None, inferred from the selected imgpoints as
        (ceil(max_x)+1, ceil(max_y)+1).
    rng_seed : int, optional
        Seed for reproducibility.
    elitism : int, optional
        Number of top individuals carried over unchanged each generation.
    tournament_k : int, optional
        Tournament size for parent selection.

    Returns
    -------
    result : dict
        {
          "best_indices": tuple[int],
          "best_rms": float,
          "K": np.ndarray (3x3),
          "dist": np.ndarray,
          "rvecs": list[np.ndarray],
          "tvecs": list[np.ndarray],
          "history": list[float],  # best RMS per generation
          "best_generation": int
        }
    """
    assert 0 < subset_size <= len(img_points), "subset_size out of range"
    assert len(ref_points) == len(img_points), "ref_points and obj_points must have the same length"
    assert population_size >= 2, "population_size must be >= 2"
    assert 0 < replace_ratio <= 1.0, "replace_ratio must be in (0,1]"
    assert mutation >= 1 and mutation <= subset_size, "mutation in [1, subset_size]"
    assert elitism >= 0 and elitism < population_size, "elitism must be < population_size"
    assert tournament_k >= 2 and tournament_k <= population_size, "tournament_k in [2, population_size]"

    rng = np.random.default_rng(rng_seed)
    n_frames = len(img_points)
    frame_indices = np.arange(n_frames)

    def eval_individual(indices_tuple):
        """Return (rms, K, dist, rvecs, tvecs). Lower rms is better."""
        idx = list(indices_tuple)
        obj = [img_points[i] for i in idx if img_points[i].shape[0] > 0]
        img = [ref_points[i] for i in idx if ref_points[i].shape[0] > 0]

        try:
            # OpenCV expects (width, height)
            ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj, img, img_size, None, None)
            rms = float(ret)
            if not np.isfinite(rms):
                rms = np.inf
            return rms, K, dist, rvecs, tvecs
        except cv2.error as e:
            print(f"OpenCV error: {e}")
            return np.inf, None, None, None, None

    def make_random_individual():
        inds = rng.choice(frame_indices, size=subset_size, replace=False)
        inds.sort()
        return tuple(inds.tolist())

    def mutate(indices_tuple):
        """Replace `mutation` positions with new, distinct indices."""
        current = list(indices_tuple)
        # positions to replace
        pos = rng.choice(subset_size, size=mutation, replace=False)
        in_set = set(current)
        available = np.array(list(set(frame_indices) - in_set))
        if available.size < mutation:
            # In degenerate cases, just return unchanged
            return indices_tuple
        new_vals = rng.choice(available, size=mutation, replace=False)
        for p, v in zip(pos, new_vals):
            current[p] = int(v)
        current = sorted(set(current))  # ensure uniqueness & order
        # If uniqueness removed too many, top up with random available
        while len(current) < subset_size:
            candidate = int(rng.choice(list(set(frame_indices) - set(current))))
            current.append(candidate)
            current = sorted(current)
        return tuple(current)

    def crossover(a, b):
        """Set-aware one-point crossover with repair."""
        a = list(a)
        b = list(b)
        cut = int(rng.integers(1, subset_size))  # in [1, subset_size-1]
        child_set = set(a[:cut]) | set(b[cut:])
        # Repair: fill up to subset_size with random, non-duplicate indices
        if len(child_set) < subset_size:
            pool = list(set(frame_indices) - child_set)
            rng.shuffle(pool)
            need = subset_size - len(child_set)
            child_set.update(pool[:need])
        child = sorted(child_set)
        if len(child) > subset_size:
            # Trim deterministically to exact size for stability
            child = child[:subset_size]
        return tuple(child)

    def tournament_select(pop, fitness, k=tournament_k):
        """Pick best of k random candidates (lower fitness is better)."""
        cand = rng.choice(len(pop), size=k, replace=False)
        best_i = min(cand, key=lambda i: fitness[i])
        return pop[best_i]

    # ---- Initialize population ----------------------------------------------

    population = []
    seen = set()
    with tqdm(total=population_size, desc="Initializing population", leave=False) as pbar:
        while len(population) < population_size:
            ind = make_random_individual()
            if ind in seen:
                continue
            population.append(ind)
            seen.add(ind)
            pbar.update(1)

    # ---- Main GA loop --------------------------------------------------------

    history = []
    best_overall = None
    best_record = {
        "best_indices": None, "best_rms": np.inf, "K": None, "dist": None, "rvecs": None, "tvecs": None,
        "history": history, "best_generation": -1
    }

    for gen in tqdm(range(generations), desc="GA generations"):
        # Evaluate population
        fitness = []
        cache = {}  # cache evaluations for identical individuals
        for ind in population:
            if ind in cache:
                fitness.append(cache[ind][0])
                continue
            rms, K, dist, rvecs, tvecs = eval_individual(ind)
            cache[ind] = (rms, K, dist, rvecs, tvecs)
            fitness.append(rms)

        # Rank population
        order = np.argsort(np.array(fitness))
        population = [population[i] for i in order]
        fitness = [fitness[i] for i in order]

        # Track best
        if fitness[0] < best_record["best_rms"]:
            rms, K, dist, rvecs, tvecs = cache[population[0]]
            best_record.update({
                "best_indices": population[0],
                "best_rms": float(fitness[0]),
                "K": K,
                "dist": dist,
                "rvecs": rvecs,
                "tvecs": tvecs,
                "best_generation": gen
            })
        history.append(float(fitness[0]))

        # Elitism: keep top N
        next_pop = population[:elitism]

        # How many to replace/create this gen
        n_replace = int(np.ceil(replace_ratio * population_size))
        n_replace = max(n_replace, population_size - elitism)  # ensure we still fill to pop size

        # Create children
        children = []
        attempts = 0
        target_children = population_size - elitism
        while len(children) < target_children and attempts < target_children * 10:
            p1 = tournament_select(population, fitness, tournament_k)
            p2 = tournament_select(population, fitness, tournament_k)
            if p1 == p2:
                attempts += 1
                continue
            child = crossover(p1, p2)
            child = mutate(child)
            if (child not in cache) and (child not in children) and (child not in next_pop):
                children.append(child)
            attempts += 1

        # If we couldn't make enough unique children, pad randomly
        while len(children) + len(next_pop) < population_size:
            ind = make_random_individual()
            if (ind not in cache) and (ind not in children) and (ind not in next_pop):
                children.append(ind)

        # Build next generation: elites + best of (children + survivors)
        # Survivors: take from the current population (after elites), if needed.
        next_pop.extend(children[: population_size - len(next_pop)])
        population = next_pop

    return best_record

```

```python
def compute_reprojection_errors(image_points, ref_points, K, dist):
    """
    Compute per-frame reprojection error (mean L2 per point) and global RMS.

    Parameters
    ----------
    groups : list[tuple[np.ndarray, np.ndarray]]
        Each element is (objpoints, imgpoints) for a checkerboard frame.
        objpoints: (N, 3) float32
        imgpoints: (N, 2) float32
    K : np.ndarray
        Camera matrix (3x3).
    dist : np.ndarray
        Distortion coefficients (length 4, 5, 8, etc.)

    Returns
    -------
    errors : np.ndarray, shape (n_frames,)
        Mean reprojection error (in pixels) per frame.
    global_rms : float
        Global RMS error across all frames.
    """
    mean_errors = []
    median_errors = []
    max_errors = []
    scatter_data = []

    for imgpoints, objpoints in tqdm(zip(image_points, ref_points), desc="Computing reprojection errors"):
        # Calibrate per-frame extrinsics (rvec, tvec) using known intrinsics
        success, rvec, tvec = cv2.solvePnP(objpoints, imgpoints, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
        if not success:
            mean_errors.append(np.nan)
            median_errors.append(np.nan)
            max_errors.append(np.nan)
            continue

        # Project 3D points into image plane
        imgpoints_proj, _ = cv2.projectPoints(objpoints, rvec, tvec, K, dist)

        # Compute per-point Euclidean distances
        diff = imgpoints_proj.squeeze() - imgpoints
        err = np.linalg.norm(diff, axis=1)
        scatter_data.extend([(x, y, e) for (e, (x, y)) in zip(err, imgpoints)])
        mean_err = np.mean(err)

        mean_errors.append(np.mean(err))
        median_errors.append(np.median(err))
        max_errors.append(np.max(err))

    return mean_errors, median_errors, max_errors, scatter_data
```

```python
def prepare_calibration(input_video, data):
    to_dir = os.path.sep.join(input_video.split(os.path.sep)[:-1])
    image_points = [np.concatenate([np.array(ref_points).reshape(-1, 1, 2),
                                    np.zeros((ref_points.shape[0], 1, 1))], axis=2).astype(np.float32)
                    for _, obj, ref_points, _ in data if len(obj) > 10]
    ref_points = [np.array(objp).astype(np.float32)
                  for _, objp, ref_points, _ in data if len(objp) > 10]

    best_record = genetic_calibration_subset(
        image_points, ref_points,
        subset_size=20,
        population_size=35,
        mutation=2,
        replace_ratio=0.25,
        generations=55,
        img_size=cropped_image.shape[::-1],
        rng_seed=42,
        elitism=2,
        tournament_k=3,
    )

    plt.figure(figsize=(15, 5))
    plt.plot(best_record["history"])
    plt.title(f"Search of spherical distortion")
    plt.xlabel("GA epoch")
    plt.ylabel("Reprojection error in pixels")
    plt.savefig(os.path.join(to_dir, f"ga-performance.png"))
    plt.show()

    mean, mede, maxe, scatter_data = compute_reprojection_errors(
        ref_points, image_points,
        best_record["K"],
        best_record["dist"]
    )
    scatter_data = np.array(scatter_data)

    plt.figure(figsize=(15, 10))
    plt.scatter(scatter_data[:, 0], scatter_data[:, 1],
                c=scatter_data[:, 2] / np.max(scatter_data[:, 2]) * 100, norm=cls.LogNorm(),
                s=scatter_data[:, 2] / np.max(scatter_data[:, 2]) * 100,
                cmap="YlGn",
                alpha=scatter_data[:, 2] / np.max(scatter_data[:, 2]) / 10)
    plt.title(f"Distribution of the error over the frame")
    plt.xlabel("x (pixel coordinate)")
    plt.ylabel("y (pixel coordinate)")
    plt.savefig(os.path.join(to_dir, f"err-distribution.png"))
    plt.show()

    plt.figure(figsize=(15, 4))
    counts, bins, patches = plt.hist(scatter_data[:, 2], bins=[0.5, 1, 1.5, 2, 2.5, 3.5, 5, 9, 14, 20, 30])
    for i, patch in enumerate(patches):
        patch.set_facecolor(plt.cm.rainbow(i / len(patches)))
    plt.yscale("log")
    plt.title(f"Reprojection error")
    plt.xlabel("Error in pixels")
    plt.ylabel("Number of points")
    plt.savefig(os.path.join(to_dir, f"reprojection-err.png"))
    plt.show()

    return {
        "transformation": best_record,
        "reprojection_error": {
            "mean_per_frame": mean,
            "median_per_frame": mede,
            "maximum_per_frame": maxe,
            "per_point": scatter_data
        }
    }
```

```python
tblr = args["checkerboard_detection"]["corner_detection_safety_margin_tblr_px"]
cropped_image = data6B2[100][0][tblr[0]:-tblr[1],tblr[2]:-tblr[3]]
```

```python
c6C1 = prepare_calibration(input_video_6C1, data6C1)
```

```python
c6B2 = prepare_calibration(input_video_6B2, data6B2)
```

```python
def distortion_displacement_in_mask(
    mask: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
) -> dict:
    """
    Measure pixel displacement caused solely by lens-distortion correction.

    Parameters
    ----------
    mask
        2D boolean or binary mask. Its coordinates refer to the original,
        distorted image.
    K
        3x3 camera matrix corresponding to the mask/image resolution.
    dist
        OpenCV distortion coefficients.

    Returns
    -------
    dict containing:
        max_displacement_px
        mean_displacement_px
        median_displacement_px
        p95_displacement_px
        max_location_xy
        displacement_map
        dx_map
        dy_map

    Notes
    -----
    The corrected points are expressed using the same K, so the result does
    not include resizing, cropping, a changed focal length, or principal-point
    displacement introduced by a new camera matrix.
    """
    if mask.ndim != 2:
        raise ValueError("mask must be a 2D array")

    K = np.asarray(K, dtype=np.float64)
    dist = np.asarray(dist, dtype=np.float64)

    if K.shape != (3, 3):
        raise ValueError(f"K must have shape (3, 3), got {K.shape}")

    valid = mask.astype(bool)
    y, x = np.nonzero(valid)

    if x.size == 0:
        raise ValueError("mask contains no selected pixels")

    # Pixel coordinates in the original distorted image.
    distorted_points = np.column_stack((x, y)).astype(np.float64)
    distorted_points = distorted_points.reshape(-1, 1, 2)

    # Corresponding ideal, undistorted pixel coordinates.
    # P=K keeps the same linear camera model.
    undistorted_points = cv2.undistortPoints(
        distorted_points,
        cameraMatrix=K,
        distCoeffs=dist,
        R=None,
        P=K,
    ).reshape(-1, 2)

    displacement_xy = undistorted_points - distorted_points.reshape(-1, 2)
    displacement = np.linalg.norm(displacement_xy, axis=1)

    max_index = int(np.argmax(displacement))
    max_x = int(x[max_index])
    max_y = int(y[max_index])

    displacement_map = np.full(mask.shape, np.nan, dtype=np.float32)
    dx_map = np.full(mask.shape, np.nan, dtype=np.float32)
    dy_map = np.full(mask.shape, np.nan, dtype=np.float32)

    displacement_map[y, x] = displacement.astype(np.float32)
    dx_map[y, x] = displacement_xy[:, 0].astype(np.float32)
    dy_map[y, x] = displacement_xy[:, 1].astype(np.float32)

    return {
        "max_displacement_px": float(displacement[max_index]),
        "mean_displacement_px": float(np.mean(displacement)),
        "median_displacement_px": float(np.median(displacement)),
        "p95_displacement_px": float(np.percentile(displacement, 95)),
        "max_location_xy": (max_x, max_y),
        "max_vector_xy_px": tuple(displacement_xy[max_index]),
        "displacement_map": displacement_map,
        "dx_map": dx_map,
        "dy_map": dy_map,
    }
```

```python
xx, yy = np.mgrid[:cropped_image.shape[1], :cropped_image.shape[0]]
circle = (xx - 1250) ** 2 + (yy - 1030) ** 2
sample_mask = (circle < 950 ** 2).T
```

```python
ddmB = distortion_displacement_in_mask(
    sample_mask,
    c6B2["transformation"]["K"],
    c6B2["transformation"]["dist"]
)
```

```python
ddmC = distortion_displacement_in_mask(
    sample_mask,
    c6C1["transformation"]["K"],
    c6C1["transformation"]["dist"]
)
```

```python
plt.imshow(ddmC["displacement_map"] - ddmB["displacement_map"], cmap="viridis")
plt.show()
```

```python
plt.imshow(ddmC["displacement_map"], cmap="viridis")
plt.show()
```

```python
plt.figure(figsize=(15, 5))
ax = plt.subplot(1,2,1)
ax.hist((ddmB["displacement_map"] - ddmC["displacement_map"]).reshape(-1), bins=100)
ax.set_title("Displacement of pixels after calibration (C1 - C2)")
ax.set_xlabel("Displacement in pixels")
ax.set_ylabel("Number of pixels")
ax = plt.subplot(1,2,2)
ax.imshow(ddmC["displacement_map"] - ddmB["displacement_map"], cmap="viridis")
ax.set_title("Displacement map")
ax.set_xticks([])
ax.set_yticks([])
plt.show()
```

```python
np.nanmax(ddmB["displacement_map"]), np.nanmax(ddmC["displacement_map"])
```

```python
plt.figure(figsize=(15, 10))
img = np.copy(data6B2[300][0]).astype(float)
img[tblr[0]:-tblr[1],tblr[2]:-tblr[3]] = img[tblr[0]:-tblr[1],tblr[2]:-tblr[3]] * (sample_mask + 1)
plt.imshow(img, cmap="gray")
plt.show()
```

```python
sample_mask.shape
```

```python
c6B2["transformation"]["dist"]
```

```python
c6C1["transformation"]["dist"]
```

```python
c6C1["transformation"]
```

```python
c6B2["transformation"]["K"]
```

```python
data6B2[150][1]
```

```python
data6B2[150][2]
```

```python

```
