"""Tests for spatial tissue graphs, Moran's I / Geary's C, LISA hotspots, and L-R signaling."""

from __future__ import annotations

import numpy as np

from cyto_manifold.container import TranscriptomeDataset
from cyto_manifold.spatial import (
    analyze_spatial_tissue,
    benjamini_hochberg,
    build_spatial_graph,
    compute_gearys_c,
    compute_local_moran,
    compute_morans_i,
    detect_spatially_variable_genes,
    score_ligand_receptor_signaling,
)
from cyto_manifold.types import LigandReceptorPair


def test_spatial_graph_modes_symmetry_and_zero_diag() -> None:
    rng = np.random.default_rng(19)
    coords = rng.uniform(0.0, 10.0, size=(36, 2))
    for mode in ("delaunay", "knn", "radius"):
        w = build_spatial_graph(coords, mode=mode, n_neighbors=6)
        assert w.shape == (36, 36)
        assert np.allclose(w, w.T, atol=1e-12)
        assert np.allclose(np.diag(w), 0.0)
        assert np.all(w >= 0.0)
        assert np.sum(w > 0.0) > 0


def test_morans_i_and_gearys_c_detect_spatial_gradient_vs_noise() -> None:
    # 6x6 regular spatial lattice
    gx, gy = np.meshgrid(np.arange(6.0), np.arange(6.0))
    coords = np.column_stack([gx.ravel(), gy.ravel()])
    w = build_spatial_graph(coords, mode="knn", n_neighbors=4)

    # Strong spatial gradient gene vs i.i.d. random noise gene
    rng = np.random.default_rng(42)
    spatial_gene = coords[:, 0] + 0.5 * coords[:, 1]
    noise_gene = rng.normal(0.0, 1.0, size=36)

    mi_sp, ei_sp, zi_sp, pi_sp = compute_morans_i(spatial_gene, w)
    gc_sp, _ = compute_gearys_c(spatial_gene, w)
    mi_ns, _, zi_ns, _ = compute_morans_i(noise_gene, w)

    assert np.isclose(ei_sp, -1.0 / 35.0)
    assert mi_sp > 0.55
    assert zi_sp > 4.0
    assert pi_sp < 1e-4
    assert gc_sp < 0.5
    assert mi_sp > mi_ns


def test_local_moran_lisa_and_bh_fdr() -> None:
    gx, gy = np.meshgrid(np.arange(5.0), np.arange(5.0))
    coords = np.column_stack([gx.ravel(), gy.ravel()])
    w = build_spatial_graph(coords, mode="knn", n_neighbors=4)
    vals = np.where((coords[:, 0] >= 3.0) & (coords[:, 1] >= 3.0), 10.0, 1.0)
    lisa = compute_local_moran(vals, w)
    assert lisa.shape == (25,)
    # Core hotspot at (4, 4) has positive Local Moran's I
    assert lisa[-1] > 0.5

    fdrs = benjamini_hochberg([0.001, 0.04, 0.5, 0.8])
    assert np.all((fdrs >= 0.0) & (fdrs <= 1.0))
    assert fdrs[0] <= fdrs[1] <= fdrs[2] <= fdrs[3]


def test_ligand_receptor_paracrine_signaling_specificity() -> None:
    # Two adjacent spatial clusters: cluster 0 expresses WNT3A, cluster 1 expresses FZD7
    gx, gy = np.meshgrid(np.arange(6.0), np.arange(6.0))
    coords = np.column_stack([gx.ravel(), gy.ravel()])
    labels = np.where(coords[:, 0] < 3.0, 0, 1).astype(np.int64)

    expr = np.zeros((36, 4), dtype=np.float64)
    expr[labels == 0, 0] = 8.0  # WNT3A
    expr[labels == 1, 1] = 8.0  # FZD7
    expr[:, 2] = coords[:, 0]   # SVG marker
    expr[:, 3] = 1.0            # constant housekeeping

    ds = TranscriptomeDataset(
        X=expr,
        gene_names=["WNT3A", "FZD7", "SPATIAL_GRAD", "ACTB"],
        spatial=coords,
        obs={"cluster": labels},
    )
    sp_res = analyze_spatial_tissue(ds, mode="delaunay", n_permutations=40, seed=7)
    assert sp_res.n_spatially_variable >= 1

    lr_hits = score_ligand_receptor_signaling(
        ds,
        sp_res.spatial_weights,
        labels,
        pairs=[LigandReceptorPair("WNT3A", "FZD7", "WNT")],
        n_permutations=50,
        fdr_threshold=0.10,
        seed=7,
    )
    assert len(lr_hits) >= 1
    top = lr_hits[0]
    assert top.sender_cluster == 0
    assert top.receiver_cluster == 1
    assert top.score > 10.0
    assert top.fdr <= 0.10

    svgs = detect_spatially_variable_genes(ds, sp_res.spatial_weights)
    assert svgs[0].morans_i > 0.4
