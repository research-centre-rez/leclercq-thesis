import math
from dataclasses import dataclass, field
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional, Set

import cv2
import numpy as np
from scipy.spatial import cKDTree


# ============================================================
# Data classes
# ============================================================

@dataclass
class CornerNode:
    idx: int
    pt: np.ndarray                     # (2,)
    black_labels: Set[int] = field(default_factory=set)
    white_labels: Set[int] = field(default_factory=set)
    ring_samples: List[Tuple[float, int, int]] = field(default_factory=list)  # (angle, color, label)
    local_score: float = 0.0
    is_synthetic: bool = False


@dataclass
class EdgeData:
    i: int
    j: int
    dist: float
    shared_black: Set[int]
    shared_white: Set[int]
    photo_score: float
    topo_score: float
    total_score: float
    dir_class: Optional[int] = None    # 0 or 1 after orientation clustering


# ============================================================
# Utility functions
# ============================================================

def _as_uint8_binary(binary: np.ndarray) -> np.ndarray:
    """
    Convert binary image to 0/255 uint8.
    """
    if binary.dtype != np.uint8:
        binary = binary.astype(np.uint8)

    vals = np.unique(binary)
    if len(vals) <= 2:
        # map min->0, max->255
        lo = vals.min()
        hi = vals.max()
        if lo == hi:
            raise ValueError("Binary image contains only one value.")
        out = np.where(binary > (lo + hi) / 2, 255, 0).astype(np.uint8)
        return out

    # fallback threshold
    _, out = cv2.threshold(binary, 127, 255, cv2.THRESH_BINARY)
    return out


def _clip_point(pt: np.ndarray, w: int, h: int) -> np.ndarray:
    x = float(np.clip(pt[0], 0, w - 1))
    y = float(np.clip(pt[1], 0, h - 1))
    return np.array([x, y], dtype=float)


def _norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def _unit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < eps:
        return np.zeros_like(v)
    return v / n


def _angle_undirected(v: np.ndarray) -> float:
    """
    Return angle in [0, pi).
    """
    a = math.atan2(v[1], v[0])
    return (a + math.pi) % math.pi


def _line_sample_points(p0: np.ndarray, p1: np.ndarray, n: int) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n)
    return (1.0 - t)[:, None] * p0[None, :] + t[:, None] * p1[None, :]


def _sample_image_nearest(img: np.ndarray, pts: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    xy = np.round(pts).astype(int)
    xy[:, 0] = np.clip(xy[:, 0], 0, w - 1)
    xy[:, 1] = np.clip(xy[:, 1], 0, h - 1)
    return img[xy[:, 1], xy[:, 0]]


def _fit_affine(ij: np.ndarray, pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit image point ~= origin + i*u + j*v
    """
    X = np.column_stack([np.ones(len(ij)), ij[:, 0], ij[:, 1]])
    A, _, _, _ = np.linalg.lstsq(X, pts, rcond=None)
    origin = A[0]
    u = A[1]
    v = A[2]
    return origin, u, v


def _connected_components(binary_mask: np.ndarray) -> np.ndarray:
    """
    Connected components on uint8 mask {0,1}.
    Returns labels with background 0.
    """
    _, labels = cv2.connectedComponents(binary_mask.astype(np.uint8), connectivity=8)
    return labels


def _merge_close_points(points: np.ndarray, radius: float) -> np.ndarray:
    """
    Simple clustering for duplicate detector outputs.
    Keeps centroid of each radius-connected group.
    """
    pts = np.asarray(points, dtype=float)
    if len(pts) == 0:
        return pts

    tree = cKDTree(pts)
    visited = np.zeros(len(pts), dtype=bool)
    merged = []

    for i in range(len(pts)):
        if visited[i]:
            continue
        cluster = tree.query_ball_point(pts[i], r=radius)
        cluster = [j for j in cluster if not visited[j]]
        visited[cluster] = True
        merged.append(pts[cluster].mean(axis=0))

    return np.asarray(merged, dtype=float)


# ============================================================
# Stage 1: cut disks around detected corners
# ============================================================

def cut_corner_disks(binary: np.ndarray, corners: np.ndarray, radius: int = 5) -> np.ndarray:
    """
    Replaces disk around each corner by neutral value 127, so black/white components
    do not join through the corner.
    """
    out = binary.copy()
    for x, y in np.round(corners).astype(int):
        cv2.circle(out, (int(x), int(y)), int(radius), 127, -1)
    return out


def compute_color_components(binary_cut: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute connected components separately for black and white.
    """
    black_mask = (binary_cut == 0).astype(np.uint8)
    white_mask = (binary_cut == 255).astype(np.uint8)

    black_labels = _connected_components(black_mask)
    white_labels = _connected_components(white_mask)
    return black_labels, white_labels


# ============================================================
# Stage 2: attach corners to surrounding black/white regions
# ============================================================

def sample_corner_ring_labels(
    black_labels: np.ndarray,
    white_labels: np.ndarray,
    center: np.ndarray,
    ring_radius: int = 7,
    n_samples: int = 48,
) -> Tuple[Set[int], Set[int], List[Tuple[float, int, int]]]:
    """
    Samples a ring around corner center and collects incident black/white region labels.

    Returns
    -------
    black_set : set of black labels > 0
    white_set : set of white labels > 0
    ring_info : list of (angle, color, label)
                color: 0 for black, 1 for white
    """
    h, w = black_labels.shape
    x0, y0 = center

    black_set = set()
    white_set = set()
    ring_info = []

    for t in np.linspace(0, 2 * np.pi, n_samples, endpoint=False):
        x = int(round(x0 + ring_radius * np.cos(t)))
        y = int(round(y0 + ring_radius * np.sin(t)))
        if x < 0 or x >= w or y < 0 or y >= h:
            continue

        b = int(black_labels[y, x])
        wlab = int(white_labels[y, x])

        # after cutting disks, each ring sample should belong to either black or white
        if b > 0:
            black_set.add(b)
            ring_info.append((float(t), 0, b))
        elif wlab > 0:
            white_set.add(wlab)
            ring_info.append((float(t), 1, wlab))

    ring_info.sort(key=lambda z: z[0])
    return black_set, white_set, ring_info


def alternation_score(ring_info: List[Tuple[float, int, int]]) -> float:
    """
    Measures whether colors alternate around the corner.
    Higher is better.
    """
    if len(ring_info) < 6:
        return -3.0

    colors = [x[1] for x in ring_info]
    changes = 0
    for i in range(len(colors)):
        if colors[i] != colors[(i + 1) % len(colors)]:
            changes += 1

    # true checkerboard inner corner should have repeated alternation
    frac = changes / len(colors)
    return 4.0 * frac - 2.0


def local_corner_score(node: CornerNode) -> float:
    """
    Conservative local score based on region incidence topology.
    """
    nb = len(node.black_labels)
    nw = len(node.white_labels)

    score = 0.0
    score += alternation_score(node.ring_samples)

    # ideal inner corner: roughly 2 black + 2 white around ring, but segmentation can split
    score -= abs(nb - 2) * 0.8
    score -= abs(nw - 2) * 0.8

    # reject obviously implausible structures
    if nb == 0 or nw == 0:
        score -= 6.0

    if nb > 4 or nw > 4:
        score -= 3.0

    return score


def build_corner_nodes(
    gray: np.ndarray,
    binary: np.ndarray,
    corners: np.ndarray,
    cut_radius: int = 5,
    ring_radius: int = 7,
    dedup_radius: Optional[float] = None,
) -> Tuple[List[CornerNode], np.ndarray, np.ndarray, np.ndarray]:
    """
    Build corner nodes from raw detections and region incidence.
    """
    binary = _as_uint8_binary(binary)
    corners = np.asarray(corners, dtype=float)

    if dedup_radius is not None and dedup_radius > 0:
        corners = _merge_close_points(corners, dedup_radius)

    binary_cut = cut_corner_disks(binary, corners, radius=cut_radius)
    black_labels, white_labels = compute_color_components(binary_cut)

    nodes = []
    for idx, pt in enumerate(corners):
        bset, wset, ring_info = sample_corner_ring_labels(
            black_labels, white_labels, pt, ring_radius=ring_radius
        )
        node = CornerNode(
            idx=idx,
            pt=np.asarray(pt, dtype=float),
            black_labels=bset,
            white_labels=wset,
            ring_samples=ring_info,
        )
        node.local_score = local_corner_score(node)
        nodes.append(node)

    n_good = sum((len(n.black_labels) > 0 and len(n.white_labels) > 0) for n in nodes)

    return nodes, binary_cut, black_labels, white_labels


# ============================================================
# Stage 3: build candidate graph from shared regions + photo validation
# ============================================================

def estimate_nominal_spacing(nodes, max_k=12, min_plausible=8.0, max_plausible=120.0):
    """
    Robust estimate of checkerboard corner spacing in pixels.

    Ignores tiny duplicate distances and uses several nearest neighbors.
    """
    pts = np.array([n.pt for n in nodes], dtype=float)
    if len(pts) < 2:
        return 1.0

    tree = cKDTree(pts)
    dists, _ = tree.query(pts, k=min(max_k + 1, len(pts)))

    # ignore self-distance in column 0
    cand = dists[:, 1:].reshape(-1)

    # reject implausibly tiny / huge distances
    cand = cand[(cand >= min_plausible) & (cand <= max_plausible)]

    if len(cand) == 0:
        raise RuntimeError("Could not estimate plausible checkerboard spacing.")

    # histogram mode is better than median when duplicates are present
    hist, edges = np.histogram(cand, bins=60, range=(min_plausible, max_plausible))
    i = np.argmax(hist)
    lo, hi = edges[i], edges[i + 1]

    # refine by median inside dominant bin neighborhood
    support = cand[(cand >= lo - 3) & (cand <= hi + 3)]
    if len(support) == 0:
        support = cand[(cand >= lo) & (cand <= hi)]

    step = float(np.median(support))
    return step


def edge_photometric_score(
    binary: np.ndarray,
    p0: np.ndarray,
    p1: np.ndarray,
    offset: float = 2.5,
    n_samples: int = 11,
) -> float:
    """
    Validate that the segment between p0,p1 separates black/white on opposite sides.

    Returns score in roughly [-1, +1].
    """
    v = p1 - p0
    L = np.linalg.norm(v)
    if L < 1e-6:
        return -1.0

    tang = v / L
    normal = np.array([-tang[1], tang[0]], dtype=float)

    pts_mid = _line_sample_points(p0, p1, n_samples)
    left_pts = pts_mid + offset * normal[None, :]
    right_pts = pts_mid - offset * normal[None, :]

    lv = _sample_image_nearest(binary, left_pts)
    rv = _sample_image_nearest(binary, right_pts)

    # Good checkerboard edge: opposite colors across the edge
    opposite = np.mean(lv != rv)
    same = np.mean(lv == rv)
    return float(opposite - same)


def build_candidate_edges(
    nodes,
    binary,
    square_size: int = 20,
    photo_offset: float = 3.0,
    require_exact_shared_regions=False,
):
    max_dist = 1.5 * square_size
    min_dist = 0.5 * square_size

    pts = np.array([n.pt for n in nodes], dtype=float)
    tree = cKDTree(pts)
    pairs = tree.query_pairs(r=max_dist)

    edges = []
    debug_counts = {
        "all_pairs_in_radius": 0,
        "passed_distance": 0,
        "passed_region_test": 0,
        "passed_photo": 0,
    }

    for i, j in pairs:
        debug_counts["all_pairs_in_radius"] += 1

        pi = nodes[i].pt
        pj = nodes[j].pt
        d = np.linalg.norm(pi - pj)

        if d < min_dist or d > max_dist:
            continue
        debug_counts["passed_distance"] += 1

        shared_black = nodes[i].black_labels & nodes[j].black_labels
        shared_white = nodes[i].white_labels & nodes[j].white_labels

        if require_exact_shared_regions:
            ok_regions = (len(shared_black) == 1 and len(shared_white) == 1)
        else:
            ok_regions = (len(shared_black) >= 1 and len(shared_white) >= 1)

        if not ok_regions:
            continue
        debug_counts["passed_region_test"] += 1

        photo = edge_photometric_score(binary, pi, pj, offset=photo_offset)
        debug_counts["passed_photo"] += 1

        dist_pen = abs(d - square_size) / max(square_size, 1e-6)
        topo = 2.0 - 1.2 * dist_pen + 0.3 * min(nodes[i].local_score, nodes[j].local_score)
        total = topo + 1.5 * photo

        edges.append(
            EdgeData(
                i=i,
                j=j,
                dist=float(d),
                shared_black=shared_black,
                shared_white=shared_white,
                photo_score=float(photo),
                topo_score=float(topo),
                total_score=float(total),
            )
        )

    return edges


# ============================================================
# Stage 4: prune weak corners / build consistent graph
# ============================================================

def build_adjacency(edges: List[EdgeData]) -> Dict[int, List[EdgeData]]:
    adj = defaultdict(list)
    for e in edges:
        adj[e.i].append(e)
        adj[e.j].append(e)
    return adj


def prune_graph(
    nodes: List[CornerNode],
    edges: List[EdgeData],
    min_edge_score: float = 0.0,
    max_degree: int = 4,
    min_local_score: float = -2.5,
    max_rounds: int = 8,
) -> Tuple[List[CornerNode], List[EdgeData], Dict[int, int]]:
    """
    Iteratively prune weak corners and oversaturated graph nodes.

    Returns
    -------
    kept_nodes
    kept_edges
    old_to_new index map
    """
    keep_nodes = {n.idx for n in nodes if n.local_score >= min_local_score}
    keep_edges = [e for e in edges if e.total_score >= min_edge_score and e.i in keep_nodes and e.j in keep_nodes]

    for _ in range(max_rounds):
        adj = defaultdict(list)
        for e in keep_edges:
            adj[e.i].append(e)
            adj[e.j].append(e)

        changed = False

        # remove weakest incident edges if degree > max_degree
        new_edges = []
        for idx in list(keep_nodes):
            incident = sorted(adj.get(idx, []), key=lambda e: e.total_score, reverse=True)
            if len(incident) > max_degree:
                allowed = set(id(x) for x in incident[:max_degree])
                for e in adj[idx][max_degree:]:
                    changed = True

        # rebuild edge list globally using degree caps
        degree_allow = {}
        for idx in keep_nodes:
            incident = sorted(adj.get(idx, []), key=lambda e: e.total_score, reverse=True)
            degree_allow[idx] = set(id(x) for x in incident[:max_degree])

        filtered = []
        for e in keep_edges:
            if id(e) in degree_allow[e.i] and id(e) in degree_allow[e.j]:
                filtered.append(e)
            else:
                changed = True
        keep_edges = filtered

        # remove isolated or too-weak nodes
        adj = build_adjacency(keep_edges)
        to_remove = set()
        for idx in keep_nodes:
            deg = len(adj.get(idx, []))
            if deg == 0:
                to_remove.add(idx)

        if to_remove:
            keep_nodes -= to_remove
            keep_edges = [e for e in keep_edges if e.i in keep_nodes and e.j in keep_nodes]
            changed = True

        if not changed:
            break

    kept_nodes = [n for n in nodes if n.idx in keep_nodes]
    old_to_new = {old.idx: new_idx for new_idx, old in enumerate(kept_nodes)}

    kept_edges = [
        EdgeData(
            i=old_to_new[e.i],
            j=old_to_new[e.j],
            dist=e.dist,
            shared_black=e.shared_black,
            shared_white=e.shared_white,
            photo_score=e.photo_score,
            topo_score=e.topo_score,
            total_score=e.total_score,
        )
        for e in keep_edges
        if e.i in old_to_new and e.j in old_to_new
    ]

    for new_idx, node in enumerate(kept_nodes):
        node.idx = new_idx

    return kept_nodes, kept_edges, old_to_new


def largest_connected_component(nodes: List[CornerNode], edges: List[EdgeData]) -> Tuple[List[CornerNode], List[EdgeData]]:
    adj = defaultdict(list)
    for e in edges:
        adj[e.i].append(e.j)
        adj[e.j].append(e.i)

    visited = set()
    comps = []

    for n in nodes:
        if n.idx in visited:
            continue
        q = deque([n.idx])
        visited.add(n.idx)
        comp = []
        while q:
            u = q.popleft()
            comp.append(u)
            for v in adj.get(u, []):
                if v not in visited:
                    visited.add(v)
                    q.append(v)
        comps.append(comp)

    if not comps:
        return [], []

    best = max(comps, key=len)
    best_set = set(best)

    nodes2 = [n for n in nodes if n.idx in best_set]
    remap = {old.idx: k for k, old in enumerate(nodes2)}
    for k, node in enumerate(nodes2):
        node.idx = k

    edges2 = []
    for e in edges:
        if e.i in best_set and e.j in best_set:
            edges2.append(
                EdgeData(
                    i=remap[e.i],
                    j=remap[e.j],
                    dist=e.dist,
                    shared_black=e.shared_black,
                    shared_white=e.shared_white,
                    photo_score=e.photo_score,
                    topo_score=e.topo_score,
                    total_score=e.total_score,
                )
            )

    return nodes2, edges2


# ============================================================
# Stage 5: classify graph edges into two direction families
# ============================================================

def cluster_edge_directions(nodes: List[CornerNode], edges: List[EdgeData]) -> Tuple[float, float]:
    """
    Find two dominant undirected angle modes.
    """
    if len(edges) < 2:
        raise RuntimeError("Not enough edges to estimate two dominant directions.")

    angles = []
    for e in edges:
        v = nodes[e.j].pt - nodes[e.i].pt
        angles.append(_angle_undirected(v))

    angles = np.asarray(angles)
    hist, bins = np.histogram(angles, bins=180, range=(0, np.pi))
    centers = 0.5 * (bins[:-1] + bins[1:])

    a1 = centers[np.argmax(hist)]

    # second direction sufficiently different from first
    diff = np.abs(((centers - a1 + np.pi / 2) % np.pi) - np.pi / 2)
    hist2 = hist.copy()
    hist2[diff < np.deg2rad(20)] = 0

    if hist2.max() <= 0:
        raise RuntimeError("Could not find second dominant direction.")

    a2 = centers[np.argmax(hist2)]
    return float(a1), float(a2)


def classify_edge_families(nodes: List[CornerNode], edges: List[EdgeData], a1: float, a2: float) -> None:
    for e in edges:
        v = nodes[e.j].pt - nodes[e.i].pt
        a = _angle_undirected(v)

        d1 = abs(((a - a1 + np.pi / 2) % np.pi) - np.pi / 2)
        d2 = abs(((a - a2 + np.pi / 2) % np.pi) - np.pi / 2)

        e.dir_class = 0 if d1 < d2 else 1


# ============================================================
# Stage 6: add missing corners by parallelogram completion
# ============================================================

def _corner_region_signature_at_point(
    pt: np.ndarray,
    black_labels: np.ndarray,
    white_labels: np.ndarray,
    ring_radius: int = 7,
) -> Tuple[Set[int], Set[int], List[Tuple[float, int, int]], float]:
    bset, wset, ring = sample_corner_ring_labels(
        black_labels, white_labels, pt, ring_radius=ring_radius
    )
    dummy = CornerNode(idx=-1, pt=pt, black_labels=bset, white_labels=wset, ring_samples=ring)
    return bset, wset, ring, local_corner_score(dummy)


def add_missing_corners(
    nodes: List[CornerNode],
    edges: List[EdgeData],
    binary: np.ndarray,
    square_size: float,
    black_labels: np.ndarray,
    white_labels: np.ndarray,
    ring_radius: int = 7,
    detect_radius: float = 0.35,
    max_additions: int = 200,
) -> Tuple[List[CornerNode], List[EdgeData]]:
    """
    Parallelogram completion:
      if A-B is one direction family and A-C the other, predict D = B + C - A.

    Validate D by:
      - no existing corner too close
      - reasonable local region topology
    """
    if len(edges) == 0:
        return nodes, edges

    pts = np.array([n.pt for n in nodes], dtype=float)
    tree = cKDTree(pts)

    adj = defaultdict(list)
    for e in edges:
        adj[e.i].append((e.j, e))
        adj[e.j].append((e.i, e))

    new_nodes = list(nodes)
    predicted = []

    for a in range(len(nodes)):
        nbs = adj.get(a, [])
        fam0 = [(b, e) for b, e in nbs if e.dir_class == 0]
        fam1 = [(b, e) for b, e in nbs if e.dir_class == 1]

        for b, eab in fam0:
            for c, eac in fam1:
                pa = nodes[a].pt
                pb = nodes[b].pt
                pc = nodes[c].pt
                pd = pb + pc - pa

                # near image?
                h, w = binary.shape
                if pd[0] < 0 or pd[0] >= w or pd[1] < 0 or pd[1] >= h:
                    continue

                # not too close to existing node
                dmin = tree.query(pd)[0]
                if dmin < detect_radius * square_size:
                    continue

                bset, wset, ring, score = _corner_region_signature_at_point(
                    pd, black_labels, white_labels, ring_radius=ring_radius
                )

                if score < -1.0:
                    continue

                predicted.append((pd, bset, wset, ring, score))
                if len(predicted) >= max_additions:
                    break
            if len(predicted) >= max_additions:
                break
        if len(predicted) >= max_additions:
            break

    for pd, bset, wset, ring, score in predicted:
        idx = len(new_nodes)
        new_nodes.append(
            CornerNode(
                idx=idx,
                pt=pd,
                black_labels=bset,
                white_labels=wset,
                ring_samples=ring,
                local_score=score,
                is_synthetic=True,
            )
        )

    # rebuild candidate edges after inserting synthetic nodes
    new_edges = build_candidate_edges(new_nodes, binary, square_size=square_size)
    return new_nodes, new_edges


# ============================================================
# Stage 7: label checkerboard graph with integer coordinates
# ============================================================

def assign_grid_coordinates(
    nodes: List[CornerNode],
    edges: List[EdgeData],
) -> Dict[int, np.ndarray]:
    """
    BFS labeling using two edge direction families.

    Returns
    -------
    coords : node_idx -> (i,j)
    """
    if len(edges) == 0:
        return {}

    adj = defaultdict(list)
    for e in edges:
        adj[e.i].append((e.j, e))
        adj[e.j].append((e.i, e))

    # seed = highest degree / strongest node
    seed = max(range(len(nodes)), key=lambda k: (len(adj.get(k, [])), nodes[k].local_score))

    coords: Dict[int, np.ndarray] = {seed: np.array([0, 0], dtype=int)}
    q = deque([seed])

    while q:
        u = q.popleft()
        pu = nodes[u].pt
        cu = coords[u]

        for v, e in adj[u]:
            pv = nodes[v].pt
            dv = pv - pu

            # directed sign selection
            if e.dir_class == 0:
                step = np.array([1, 0], dtype=int) if dv[0] >= 0 else np.array([-1, 0], dtype=int)
            else:
                step = np.array([0, 1], dtype=int) if dv[1] >= 0 else np.array([0, -1], dtype=int)

            cv = cu + step

            if v not in coords:
                coords[v] = cv
                q.append(v)
            else:
                # contradiction: keep existing if conflict is small; otherwise ignore this edge
                pass

    # normalize indices to start near zero
    if coords:
        all_ij = np.stack(list(coords.values()), axis=0)
        offset = all_ij.min(axis=0)
        for k in list(coords.keys()):
            coords[k] = coords[k] - offset

    return coords


# ============================================================
# Stage 8: consistency cleanup after labeling
# ============================================================

def deduplicate_same_grid(nodes: List[CornerNode], coords: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
    """
    If multiple nodes got same (i,j), keep best-scored one.
    """
    best = {}
    for idx, ij in coords.items():
        key = tuple(int(x) for x in ij)
        score = nodes[idx].local_score + (0.3 if not nodes[idx].is_synthetic else 0.0)
        if key not in best or score > best[key][1]:
            best[key] = (idx, score)

    out = {}
    for key, (idx, _) in best.items():
        out[idx] = np.array(key, dtype=int)
    return out


def remove_inconsistent_edges(nodes: List[CornerNode], edges: List[EdgeData], coords: Dict[int, np.ndarray]) -> List[EdgeData]:
    """
    Keep only edges consistent with one-step grid motion.
    """
    kept = []
    for e in edges:
        if e.i not in coords or e.j not in coords:
            continue
        dij = coords[e.j] - coords[e.i]
        ok = (
            (abs(dij[0]) == 1 and dij[1] == 0 and e.dir_class == 0) or
            (abs(dij[1]) == 1 and dij[0] == 0 and e.dir_class == 1)
        )
        if ok:
            kept.append(e)
    return kept


# ============================================================
# Final API
# ============================================================

def reconstruct_checkerboard_topology(
    gray: np.ndarray,
    binary: np.ndarray,
    corners: np.ndarray,
    square_size: float = 1.0,
    cut_radius: int = 5,
    ring_radius: int = 7,
    dedup_radius: Optional[float] = None,
    enable_missing_corner_completion: bool = True,
    verbose: bool = False,
):
    """
    Full topology-based checkerboard reconstruction.

    Parameters
    ----------
    gray : HxW uint8 or float image
        Original grayscale image. Currently used only as optional future cue;
        binary and topology drive the reconstruction.
    binary : HxW uint8
        Thresholded black/white image.
    corners : Nx2 float array
        Detected corners (may contain extras and duplicates, may miss some).
    square_size : float
        Grid spacing in returned obj_points.
    cut_radius : int
        Radius for cutting around each corner in the binary image.
    ring_radius : int
        Radius for corner-region incidence probing.
    dedup_radius : float or None
        Optional radius for merging near-duplicate corner detections before topology.
    enable_missing_corner_completion : bool
        Whether to insert missing corners using parallelogram completion.
    verbose : bool
        Print debug info.

    Returns
    -------
    ref_points : (M,2) float
        Image coordinates of accepted corners.
    obj_points : (M,2) float
        Corresponding checkerboard lattice coordinates in 2D.
    debug : dict
        Intermediate results.
    """
    if gray.ndim != 2:
        raise ValueError("gray must be a single-channel image")
    binary = _as_uint8_binary(binary)
    corners = np.asarray(corners, dtype=float)

    # ----------------------------
    # 1) Build initial nodes
    # ----------------------------
    nodes, binary_cut, black_labels, white_labels = build_corner_nodes(
        gray=gray,
        binary=binary,
        corners=corners,
        cut_radius=cut_radius,
        ring_radius=ring_radius,
        dedup_radius=dedup_radius,
    )

    if verbose:
        print(f"[1] initial corners: {len(nodes)}")

    if len(nodes) < 4:
        raise RuntimeError("Too few corners after preprocessing.")

    # ----------------------------
    # 2) Candidate edges
    # ----------------------------
    edges = build_candidate_edges(nodes, binary, square_size, photo_offset=3, require_exact_shared_regions=False)

    if verbose:
        print(f"[2] candidate edges: {len(edges)}")

    if len(edges) == 0:
        raise RuntimeError("No candidate topology edges found.")

    # ----------------------------
    # 3) Prune graph
    # ----------------------------
    nodes, edges, _ = prune_graph(
        nodes,
        edges,
        min_edge_score=1,
        max_degree=6,
        min_local_score=-2.7,
        max_rounds=5,
    )
    nodes, edges = largest_connected_component(nodes, edges)

    if verbose:
        print(f"[3] pruned corners: {len(nodes)}, pruned edges: {len(edges)}")

    if len(nodes) < 4 or len(edges) < 3:
        raise RuntimeError("Graph too small after pruning.")

    # ----------------------------
    # 4) Classify directions
    # ----------------------------
    a1, a2 = cluster_edge_directions(nodes, edges)
    classify_edge_families(nodes, edges, a1, a2)

    # ----------------------------
    # 5) Add missing corners
    # ----------------------------
    if enable_missing_corner_completion:
        nodes, edges = add_missing_corners(
            nodes=nodes,
            edges=edges,
            binary=binary,
            square_size=square_size,
            black_labels=black_labels,
            white_labels=white_labels,
            ring_radius=ring_radius,
        )
        nodes, edges, _ = prune_graph(nodes, edges)
        nodes, edges = largest_connected_component(nodes, edges)

        if len(edges) >= 2:
            a1, a2 = cluster_edge_directions(nodes, edges)
            classify_edge_families(nodes, edges, a1, a2)

        if verbose:
            n_syn = sum(n.is_synthetic for n in nodes)
            print(f"[4] after missing-corner completion: corners={len(nodes)}, edges={len(edges)}, synthetic={n_syn}")

    # ----------------------------
    # 6) Label graph with integer grid coords
    # ----------------------------
    coords = assign_grid_coordinates(nodes, edges)
    coords = deduplicate_same_grid(nodes, coords)
    edges = remove_inconsistent_edges(nodes, edges, coords)

    if verbose:
        print(f"[5] labeled nodes: {len(coords)}, consistent edges: {len(edges)}")

    if len(coords) < 4:
        raise RuntimeError("Failed to label enough checkerboard corners.")

    # ----------------------------
    # 7) Final stable ordering
    # ----------------------------
    items = sorted(coords.items(), key=lambda kv: (kv[1][1], kv[1][0]))  # sort by j, then i

    ref_points = np.array([nodes[idx].pt for idx, _ in items], dtype=float)
    ij = np.array([coord for _, coord in items], dtype=int)
    obj_points = ij.astype(float) * float(square_size)

    # ----------------------------
    # 8) Optional affine fit for diagnostics
    # ----------------------------
    origin, u, v = _fit_affine(ij.astype(float), ref_points)

    debug = {
        "nodes": nodes,
        "edges": edges,
        "coords": coords,
        "binary_cut": binary_cut,
        "black_labels": black_labels,
        "white_labels": white_labels,
        "affine_origin": origin,
        "affine_u": u,
        "affine_v": v,
        "angles": (a1, a2),
    }

    return ref_points, obj_points, debug


# ============================================================
# Visualization helpers
# ============================================================

def draw_reconstruction(
    gray: np.ndarray,
    ref_points: np.ndarray,
    obj_points: np.ndarray,
    debug: dict,
    point_radius: int = 4,
) -> np.ndarray:
    """
    Visualize reconstructed graph and labels.
    """
    vis = cv2.cvtColor(gray.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    nodes = debug["nodes"]
    edges = debug["edges"]
    coords = debug["coords"]

    # edges
    for e in edges:
        if e.i in coords and e.j in coords:
            p0 = tuple(np.round(nodes[e.i].pt).astype(int))
            p1 = tuple(np.round(nodes[e.j].pt).astype(int))
            color = (0, 255, 0) if e.dir_class == 0 else (255, 0, 0)
            cv2.line(vis, p0, p1, color, 1, cv2.LINE_AA)

    # points + labels
    for idx, ij in coords.items():
        p = tuple(np.round(nodes[idx].pt).astype(int))
        col = (0, 255, 255) if not nodes[idx].is_synthetic else (0, 128, 255)
        cv2.circle(vis, p, point_radius, col, -1, cv2.LINE_AA)
        cv2.putText(
            vis,
            f"{int(ij[0])},{int(ij[1])}",
            (p[0] + 4, p[1] - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            3.35,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return vis


# ============================================================
# Example usage
# ============================================================

if __name__ == "__main__":
    # --------------------------------------------------------
    # Synthetic demo image
    # --------------------------------------------------------
    H, W = 500, 700
    gray = np.full((H, W), 180, dtype=np.uint8)

    # draw simple checkerboard-like patch
    binary = np.full((H, W), 255, dtype=np.uint8)
    x0, y0 = 120, 80
    sx, sy = 36, 34
    nx, ny = 10, 8

    for j in range(ny):
        for i in range(nx):
            x1 = x0 + i * sx
            y1 = y0 + j * sy
            x2 = x1 + sx
            y2 = y1 + sy
            color = 0 if (i + j) % 2 == 0 else 255
            cv2.rectangle(binary, (x1, y1), (x2, y2), int(color), -1)

    gray = binary.copy()

    # create noisy corner detections
    corners = []
    for j in range(1, ny):
        for i in range(1, nx):
            pt = np.array([x0 + i * sx, y0 + j * sy], dtype=float)
            pt += np.random.normal(scale=1.0, size=2)
            corners.append(pt)

    corners = np.array(corners, dtype=float)

    # remove a few corners (missing)
    keep_mask = np.ones(len(corners), dtype=bool)
    keep_mask[[3, 14, 25]] = False
    corners = corners[keep_mask]

    # add extra false corners
    extras = np.array([
        [100, 100],
        [400, 180],
        [500, 350],
        [250, 250],
    ], dtype=float)

    # add duplicates
    dup = corners[:4] + np.random.normal(scale=1.5, size=(4, 2))

    corners = np.vstack([corners, extras, dup])

    ref_points, obj_points, debug = reconstruct_checkerboard_topology(
        gray=gray,
        binary=binary,
        corners=corners,
        square_size=1.0,
        cut_radius=4,
        ring_radius=6,
        dedup_radius=4.0,
        enable_missing_corner_completion=True,
        verbose=True,
    )

    print("ref_points shape:", ref_points.shape)
    print("obj_points shape:", obj_points.shape)

    vis = draw_reconstruction(gray, ref_points, obj_points, debug)
    cv2.imwrite("checkerboard_topology_reconstruction.png", vis)
    print("Saved visualization to checkerboard_topology_reconstruction.png")