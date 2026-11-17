"""Tests for GRN mutual information/CLR/AUCell inference and DE/GSEA/drivers."""

from __future__ import annotations

import numpy as np

from cyto_manifold.container import TranscriptomeDataset
from cyto_manifold.differential import (
    detect_trajectory_drivers,
    rank_genes_groups,
    run_gsea_enrichment,
    welch_t_test_gene,
    wilcoxon_rank_sum_gene,
)
from cyto_manifold.grn import (
    compute_aucell_scores,
    compute_clr_matrix,
    compute_mutual_information,
    infer_gene_regulatory_network,
)


def test_mutual_information_and_clr_detect_nonlinear_dependency() -> None:
    rng = np.random.default_rng(10)
    x = rng.uniform(-2.0, 2.0, size=120)
    y_dep = x ** 2 + rng.normal(0.0, 0.1, size=120)
    y_indep = rng.normal(0.0, 1.0, size=120)

    mi_dep = compute_mutual_information(x, y_dep, n_bins=8)
    mi_ind = compute_mutual_information(x, y_indep, n_bins=8)
    assert mi_dep > mi_ind + 0.3

    mi_mat = np.array([[mi_dep, mi_ind, 0.05, 0.03], [0.04, 0.05, 0.4, 0.02]])
    clr = compute_clr_matrix(mi_mat)
    assert clr[0, 0] > clr[0, 1]


def test_grn_inference_and_aucell_scoring() -> None:
    rng = np.random.default_rng(22)
    n_cells = 90
    tf_sox2 = np.maximum(np.concatenate([rng.normal(4.0, 0.4, 45), rng.normal(0.5, 0.2, 45)]), 0.0)
    tf_snai1 = np.maximum(np.concatenate([rng.normal(0.5, 0.2, 45), rng.normal(4.0, 0.4, 45)]), 0.0)

    g_lgr5 = np.maximum(1.2 * tf_sox2 + rng.normal(0.0, 0.3, n_cells), 0.0)
    g_rspo3 = np.maximum(0.9 * tf_sox2 + rng.normal(0.0, 0.3, n_cells), 0.0)
    g_fn1 = np.maximum(1.1 * tf_snai1 + rng.normal(0.0, 0.3, n_cells), 0.0)
    g_col1 = np.maximum(1.0 * tf_snai1 + rng.normal(0.0, 0.3, n_cells), 0.0)
    noise_genes = rng.uniform(0.5, 1.5, size=(n_cells, 6))

    expr = np.column_stack([tf_sox2, tf_snai1, g_lgr5, g_rspo3, g_fn1, g_col1, noise_genes])
    names = ["SOX2", "SNAI1", "LGR5", "RSPO3", "FN1", "COL1A1", "G0", "G1", "G2", "G3", "G4", "G5"]
    ds = TranscriptomeDataset(X=expr, gene_names=names, spliced=expr, unspliced=expr * 0.6)

    grn = infer_gene_regulatory_network(ds, tf_names=["SOX2", "SNAI1"], clr_threshold=0.5)
    assert len(grn.edges) >= 3
    edge_pairs = {(e.tf_name, e.target_name) for e in grn.edges}
    assert ("SOX2", "LGR5") in edge_pairs or ("SOX2", "RSPO3") in edge_pairs

    auc, tfs = compute_aucell_scores(expr, names, {"SOX2": ["LGR5", "RSPO3"], "SNAI1": ["FN1", "COL1A1"]})
    assert auc.shape == (n_cells, 2)
    sox_col = tfs.index("SOX2")
    assert float(np.mean(auc[:45, sox_col])) > float(np.mean(auc[45:, sox_col]))


def test_differential_expression_gsea_and_trajectory_drivers() -> None:
    rng = np.random.default_rng(31)
    n_cells = 60
    labels = np.array([0] * 30 + [1] * 30, dtype=np.int64)
    pt = np.linspace(0.0, 1.0, n_cells)

    # Cluster 0 enriched in STEMNESS genes; Cluster 1 enriched in EMT genes
    stem_block = np.vstack([rng.normal(4.5, 0.3, (30, 5)), rng.normal(0.3, 0.2, (30, 5))])
    emt_block = np.vstack([rng.normal(0.3, 0.2, (30, 5)), rng.normal(4.5, 0.3, (30, 5))])
    house = rng.normal(2.0, 0.2, (60, 4))
    expr = np.maximum(np.hstack([stem_block, emt_block, house]), 0.0)
    names = [
        "SOX2", "NANOG", "OCT4", "LGR5", "RSPO3",
        "SNAI1", "ZEB1", "TWIST1", "FN1", "COL1A1",
        "ACTB", "GAPDH", "B2M", "TBP",
    ]
    ds = TranscriptomeDataset(X=expr, gene_names=names, obs={"cluster": labels})

    z_w, p_w = wilcoxon_rank_sum_gene(expr[:30, 0], expr[30:, 0])
    t_s, p_t = welch_t_test_gene(expr[:30, 0], expr[30:, 0])
    assert z_w > 4.0 and p_w < 1e-4
    assert t_s > 10.0 and p_t < 1e-4

    de_wilcox = rank_genes_groups(ds, labels=labels, method="wilcoxon")
    de_ttest = rank_genes_groups(ds, labels=labels, method="t-test")
    assert any(d.gene_name == "SOX2" and d.cluster == 0 for d in de_wilcox)
    assert any(d.gene_name == "SNAI1" and d.cluster == 1 for d in de_ttest)

    gsea = run_gsea_enrichment(ds, labels=labels, n_permutations=40, seed=31)
    assert len(gsea) >= 2
    stem_c0 = [g for g in gsea if g.cluster == 0 and g.pathway_name == "STEMNESS_PLURIPOTENCY"]
    assert len(stem_c0) == 1 and stem_c0[0].nes > 1.0

    drivers = detect_trajectory_drivers(ds, pt, top_k=5)
    assert len(drivers) == 5
    assert abs(drivers[0][1]) > 0.7
