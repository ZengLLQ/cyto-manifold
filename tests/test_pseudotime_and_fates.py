"""Tests for Diffusion Maps, DPT, PAGA connectivity, and absorbing Markov fate probabilities."""

from __future__ import annotations

import numpy as np

from cyto_manifold.clustering import cluster_dataset
from cyto_manifold.container import TranscriptomeDataset
from cyto_manifold.manifold import build_knn_graph
from cyto_manifold.pseudotime import (
    compute_absorbing_fates,
    compute_diffusion_map,
    compute_diffusion_pseudotime,
    compute_paga_connectivity,
    infer_trajectory_and_fates,
)


def test_diffusion_map_and_dpt_linear_ordering() -> None:
    # 1D chain manifold embedded in 4D
    t = np.linspace(0.0, 1.0, 45)
    pts = np.column_stack([t, t ** 2, np.sin(t), np.cos(t)])
    graph = build_knn_graph(pts, n_neighbors=6)
    dc, lambdas, t_markov = compute_diffusion_map(graph.connectivities, n_components=5)

    assert dc.shape == (45, 5)
    assert np.allclose(np.sum(t_markov, axis=1), 1.0, atol=1e-9)
    dpt = compute_diffusion_pseudotime(dc, lambdas, root_cell=0)
    assert dpt[0] == 0.0
    assert np.all((dpt >= 0.0) & (dpt <= 1.0 + 1e-12))
    assert float(np.corrcoef(t, dpt)[0, 1]) > 0.92


def test_absorbing_fates_exact_markov_conservation() -> None:
    # Bifurcating 5-state chain: 0 -> 1 -> (2, 3), 2 absorbing, 3 absorbing
    t_mat = np.array(
        [
            [0.1, 0.9, 0.0, 0.0],
            [0.0, 0.2, 0.4, 0.4],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    term = {0: np.array([2], dtype=np.int64), 1: np.array([3], dtype=np.int64)}
    fates, entropy = compute_absorbing_fates(t_mat, term)
    assert np.allclose(np.sum(fates, axis=1), 1.0, atol=1e-9)
    # Symmetric branch means cell 0 & 1 have 50/50 fate probability and max entropy = 1.0
    assert np.allclose(fates[0], [0.5, 0.5], atol=1e-5)
    assert np.allclose(fates[1], [0.5, 0.5], atol=1e-5)
    assert np.isclose(entropy[0], 1.0, atol=1e-5)
    assert np.isclose(entropy[2], 0.0, atol=1e-5)


def test_paga_and_infer_trajectory_and_fates_bifurcation() -> None:
    rng = np.random.default_rng(64)
    # Build Y-shaped bifurcation in 4D
    stem = np.column_stack([np.linspace(0.0, 1.0, 20), np.zeros(20)])
    arm_a = np.column_stack([np.linspace(1.0, 2.5, 20), np.linspace(0.1, 1.5, 20)])
    arm_b = np.column_stack([np.linspace(1.0, 2.5, 20), np.linspace(-0.1, -1.5, 20)])
    pts = np.vstack([stem, arm_a, arm_b]) + rng.normal(0.0, 0.04, size=(60, 2))

    ds = TranscriptomeDataset(X=np.abs(pts) + 1.0)
    ds.set_obsm("X_pca", pts)
    graph = build_knn_graph(pts, n_neighbors=8)
    ds.set_obsp("connectivities", graph.connectivities)
    clust = cluster_dataset(ds, resolution=0.7, seed=64)

    res = infer_trajectory_and_fates(ds, root_cell=0, n_terminal_states=2)
    assert np.allclose(np.sum(res.fate_probabilities, axis=1), 1.0, atol=1e-8)
    assert res.paga_connectivity.shape[0] == clust.n_clusters
    assert np.allclose(res.paga_connectivity, res.paga_connectivity.T, atol=1e-12)
    paga, _ = compute_paga_connectivity(graph.connectivities, clust.labels)
    assert np.all((paga >= 0.0) & (paga <= 1.0))
