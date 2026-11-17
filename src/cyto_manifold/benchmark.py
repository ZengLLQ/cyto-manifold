"""Synthetic multi-lineage spatial tissue generator, ground-truth evaluation, and sensitivity.

Generates realistic single-cell and spatial transcriptomics benchmarks with
known ground truth across all biological modalities:

1. Branching differentiation trajectory:
     - Lineage 0 (Progenitor/Stem crypt niche) -> Lineage 1 (Epithelial) & Lineage 2 (Mesenchymal)
     - Plus spatially localized Lineage 3 (Immune/Endothelial niche)
2. Analytical RNA velocity splicing kinetics ``(u(t), s(t))`` sampled from
   Negative Binomial UMI noise around exact ODE trajectories.
3. 2D spatial tissue architecture with concentric crypt/stroma/immune zones,
   spatially variable genes (SVGs), and localized Ligand-Receptor hotspots
   (e.g., ``WNT3A -> FZD7``, ``CXCL12 -> CXCR4``, ``VEGFA -> KDR``).
4. Optional multi-batch technical capture-efficiency & gene-dropout shifts.
5. Quantitative recovery benchmarks (ARI, NMI, Spearman pseudotime, velocity
   cosine accuracy, SVG recall/precision, fate purity) and OAT sensitivity.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from scipy.stats import spearmanr

from cyto_manifold.clustering import (
    adjusted_rand_index,
    cluster_dataset,
    normalized_mutual_information,
)
from cyto_manifold.container import TranscriptomeDataset
from cyto_manifold.manifold import embed_manifold, magic_diffusion_impute
from cyto_manifold.preprocessing import (
    compute_pca,
    compute_qc_metrics,
    filter_and_normalize,
    integrate_batches_harmony,
    select_highly_variable_genes,
)
from cyto_manifold.pseudotime import infer_trajectory_and_fates
from cyto_manifold.spatial import analyze_spatial_tissue
from cyto_manifold.types import BenchmarkReport, FloatArray, TissueBenchmarkSpec
from cyto_manifold.velocity import analytical_splicing_trajectory, estimate_rna_velocity

PANEL_GENES: Tuple[str, ...] = (
    # Stem / Progenitor & WNT
    "SOX2", "NANOG", "OCT4", "LGR5", "RSPO3", "WNT3A", "FZD7", "LEF1", "TCF7", "MYC",
    # Epithelial branch
    "CDH1", "EPCAM", "KRT18", "KRT19", "EGF", "EGFR", "AREG", "ERBB2", "GATA3", "HES1",
    # Mesenchymal / TGFb / ECM branch
    "SNAI1", "ZEB1", "TWIST1", "FN1", "COL1A1", "ITGB1", "ITGA5", "TGFB1", "TGFBR1", "SMAD3",
    "PDGFA", "PDGFRA", "VIM", "SERPINE1",
    # Vascular / Niche / Immune
    "VEGFA", "VEGFB", "KDR", "FLT1", "HIF1A", "FGF2", "FGFR1", "ETS1",
    "CXCL12", "CXCR4", "CCL2", "CCR2", "IL6", "IL6R", "TNF", "TNFRSF1A", "CSF1", "CSF1R",
    "NFKB1", "IRF8", "SPI1", "DLL1", "NOTCH1", "JAG1", "NOTCH2", "BMP4", "BMPR1A",
    # Proliferation & checkpoint
    "E2F1", "MKI67", "CDK1", "PCNA", "TOP2A", "CCNB1", "CD274", "PDCD1", "LGALS9", "HAVCR2",
    # Housekeeping / non-spatial noise genes
    "ACTB", "GAPDH", "B2M", "RPL13A", "EEF1A1", "HPRT1", "TBP", "YWHAZ", "UBC",
)


def _sample_neg_binom(rng: np.random.Generator, mu: FloatArray, overdispersion: float) -> FloatArray:
    """Sample Negative Binomial counts with mean ``mu`` and dispersion ``theta``."""
    theta = max(1.0 / max(overdispersion, 1e-4), 1.0)
    mu_safe = np.maximum(mu, 1e-6)
    gamma_rate = rng.gamma(shape=theta, scale=mu_safe / theta)
    counts = rng.poisson(np.clip(gamma_rate, 0.0, 1e6)).astype(np.float64)
    return counts


def generate_synthetic_tissue(spec: TissueBenchmarkSpec) -> TranscriptomeDataset:
    """Generate a synthetic multi-lineage spatial & splicing-kinetic tissue dataset."""
    rng = np.random.default_rng(spec.seed)
    n_cells = spec.n_cells
    n_genes = max(spec.n_genes, 40)

    if n_genes <= len(PANEL_GENES):
        gene_names = list(PANEL_GENES[:n_genes])
    else:
        extra = [f"GENE_{j:04d}" for j in range(n_genes - len(PANEL_GENES))]
        gene_names = list(PANEL_GENES) + extra

    # Assign cells to 4 distinct lineages along branching pseudotime
    n_stem = n_cells // 4
    n_epi = n_cells // 4
    n_mes = n_cells // 4
    n_imm = n_cells - (n_stem + n_epi + n_mes)

    true_clusters = np.concatenate(
        [
            np.zeros(n_stem, dtype=np.int64),
            np.ones(n_epi, dtype=np.int64),
            np.full(n_mes, 2, dtype=np.int64),
            np.full(n_imm, 3, dtype=np.int64),
        ]
    )
    true_pseudotime = np.concatenate(
        [
            rng.uniform(0.02, 0.30, size=n_stem),
            rng.uniform(0.35, 0.98, size=n_epi),
            rng.uniform(0.35, 0.98, size=n_mes),
            rng.uniform(0.55, 0.95, size=n_imm),
        ]
    )

    # 2D spatial tissue architecture (crypt core -> radial branches + immune pocket)
    spatial = np.zeros((n_cells, 2), dtype=np.float64)
    # Cluster 0: central stem crypt niche
    r0 = rng.uniform(0.0, 2.4, size=n_stem)
    ang0 = rng.uniform(0.0, 2.0 * np.pi, size=n_stem)
    spatial[:n_stem, 0] = r0 * np.cos(ang0)
    spatial[:n_stem, 1] = r0 * np.sin(ang0)

    # Cluster 1: upper-right epithelial lobe
    s1 = slice(n_stem, n_stem + n_epi)
    r1 = rng.uniform(2.5, 6.2, size=n_epi)
    ang1 = rng.uniform(-0.35 * np.pi, 0.45 * np.pi, size=n_epi)
    spatial[s1, 0] = r1 * np.cos(ang1) + 1.8
    spatial[s1, 1] = r1 * np.sin(ang1) + 1.2

    # Cluster 2: lower-left mesenchymal stroma
    s2 = slice(n_stem + n_epi, n_stem + n_epi + n_mes)
    r2 = rng.uniform(2.5, 6.2, size=n_mes)
    ang2 = rng.uniform(0.65 * np.pi, 1.45 * np.pi, size=n_mes)
    spatial[s2, 0] = r2 * np.cos(ang2) - 1.8
    spatial[s2, 1] = r2 * np.sin(ang2) - 1.2

    # Cluster 3: localized immune / endothelial boundary hotspot
    s3 = slice(n_stem + n_epi + n_mes, n_cells)
    spatial[s3, 0] = rng.normal(2.2, 0.9, size=n_imm)
    spatial[s3, 1] = rng.normal(-2.6, 0.9, size=n_imm)

    # Build true splicing kinetics (u, s, velocity) per gene
    u_true = np.zeros((n_cells, n_genes), dtype=np.float64)
    s_true = np.zeros((n_cells, n_genes), dtype=np.float64)
    vel_true = np.zeros((n_cells, n_genes), dtype=np.float64)
    is_true_svg = np.zeros(n_genes, dtype=bool)

    for j, g_name in enumerate(gene_names):
        beta = float(rng.uniform(0.8, 1.4))
        gamma = float(rng.uniform(0.4, 1.1))
        alpha_base = float(rng.uniform(12.0, 28.0))

        # Assign lineage/spatial specificity by gene family
        if j < 10:
            # Stem / WNT genes: active early (cluster 0) and repressing later
            t_local = true_pseudotime * 5.2
            u_g, s_g, _ = analytical_splicing_trajectory(t_local, alpha_base, beta, gamma, t_switch=1.2)
            # Boost in stem niche
            niche_boost = np.where(true_clusters == 0, 1.8, 0.45)
            u_g *= niche_boost
            s_g *= niche_boost
            is_true_svg[j] = True
        elif j < 20:
            # Epithelial branch induction
            t_local = np.where(true_clusters == 1, true_pseudotime * 4.5, true_pseudotime * 0.8)
            u_g, s_g, _ = analytical_splicing_trajectory(t_local, alpha_base, beta, gamma, t_switch=3.6)
            mask = np.where(true_clusters == 1, 2.0, np.where(true_clusters == 0, 0.9, 0.25))
            u_g *= mask
            s_g *= mask
            is_true_svg[j] = True
        elif j < 34:
            # Mesenchymal / TGFb / ECM branch induction
            t_local = np.where(true_clusters == 2, true_pseudotime * 4.5, true_pseudotime * 0.7)
            u_g, s_g, _ = analytical_splicing_trajectory(t_local, alpha_base, beta, gamma, t_switch=3.5)
            mask = np.where(true_clusters == 2, 2.1, np.where(true_clusters == 0, 0.8, 0.25))
            u_g *= mask
            s_g *= mask
            is_true_svg[j] = True
        elif j < 61:
            # Immune / Vascular / Paracrine niche
            t_local = np.where(true_clusters == 3, true_pseudotime * 4.2, true_pseudotime * 0.9)
            u_g, s_g, _ = analytical_splicing_trajectory(t_local, alpha_base, beta, gamma, t_switch=3.2)
            mask = np.where(true_clusters == 3, 2.3, 0.35)
            u_g *= mask
            s_g *= mask
            is_true_svg[j] = True
        else:
            # Ubiquitous housekeeping / background noise genes
            u_g = rng.uniform(2.0, 5.0, size=n_cells)
            s_g = (beta / gamma) * u_g + rng.normal(0.0, 0.3, size=n_cells)
            s_g = np.maximum(s_g, 0.5)

        u_true[:, j] = u_g
        s_true[:, j] = s_g
        vel_true[:, j] = beta * u_g - gamma * s_g

    # Batch assignment & technical shift (preserving biology)
    batch_ids = [f"BATCH_{(i % max(1, spec.n_batches)) + 1}" for i in range(n_cells)]
    rng.shuffle(batch_ids)
    batch_factor = np.ones((n_cells, n_genes), dtype=np.float64)
    if spec.n_batches > 1:
        gene_batch_bias = rng.normal(0.0, spec.batch_effect_strength, size=(spec.n_batches, n_genes))
        for b_idx in range(spec.n_batches):
            b_name = f"BATCH_{b_idx + 1}"
            b_mask = np.array([b == b_name for b in batch_ids], dtype=bool)
            batch_factor[b_mask] = np.exp(np.clip(gene_batch_bias[b_idx], -1.2, 1.2))[None, :]

    s_counts = _sample_neg_binom(rng, s_true * batch_factor, spec.noise_overdispersion)
    u_counts = _sample_neg_binom(rng, u_true * batch_factor, spec.noise_overdispersion)
    total_x = s_counts + u_counts

    ds = TranscriptomeDataset(
        X=total_x,
        cell_ids=[f"SPOT_{i:04d}" for i in range(n_cells)],
        gene_names=gene_names,
        spliced=s_counts,
        unspliced=u_counts,
        spatial=spatial,
        obs={
            "true_cluster": true_clusters,
            "true_pseudotime": true_pseudotime,
            "batch": np.asarray(batch_ids),
        },
        var={
            "true_svg": is_true_svg,
        },
        uns={
            "benchmark_spec": {
                "n_cells": n_cells,
                "n_genes": n_genes,
                "n_batches": spec.n_batches,
                "seed": spec.seed,
            }
        },
    )
    ds.set_layer("true_velocity", vel_true, allow_negative=True)
    return ds


def run_benchmark_audit(spec: TissueBenchmarkSpec) -> Tuple[TranscriptomeDataset, BenchmarkReport]:
    """Run full pipeline on synthetic ground-truth tissue and score recovery."""
    ds_raw = generate_synthetic_tissue(spec)
    qc = compute_qc_metrics(
        ds_raw,
        min_genes_per_cell=5,
        min_cells_per_gene=2,
        max_doublet_score=0.98,
        seed=spec.seed,
    )
    ds = filter_and_normalize(ds_raw, qc=qc)
    select_highly_variable_genes(ds, n_top_genes=min(45, ds.n_genes))
    pca = compute_pca(ds, n_components=min(18, ds.n_genes - 1, ds.n_cells - 1))
    if spec.n_batches > 1 and "batch" in ds.obs:
        pca = integrate_batches_harmony(pca, ds.obs["batch"].tolist(), seed=spec.seed)
        ds.set_obsm("X_pca", pca.coordinates)

    graph, _ = embed_manifold(ds, n_neighbors=14, seed=spec.seed)
    magic_diffusion_impute(ds, graph, t_steps=3)
    clust = cluster_dataset(ds, resolution=0.75, method="leiden", seed=spec.seed)
    vel = estimate_rna_velocity(ds, mode="dynamical")
    pt = infer_trajectory_and_fates(ds)
    sp = analyze_spatial_tissue(ds, mode="delaunay", n_permutations=40, seed=spec.seed)

    true_clust = np.asarray(ds.obs["true_cluster"], dtype=np.int64)
    true_pt = np.asarray(ds.obs["true_pseudotime"], dtype=np.float64)
    true_vel = ds.layers["true_velocity"]
    true_svg = np.asarray(ds.var["true_svg"], dtype=bool)

    ari = adjusted_rand_index(true_clust, clust.labels)
    nmi = normalized_mutual_information(true_clust, clust.labels)

    rho, _ = spearmanr(true_pt, pt.dpt_pseudotime)
    pt_spearman = 0.0 if not np.isfinite(rho) else float(rho)

    # Cosine agreement between estimated and true velocity per cell
    v_est = vel.velocities
    est_norm = np.linalg.norm(v_est, axis=1)
    true_norm = np.linalg.norm(true_vel, axis=1)
    valid = (est_norm > 1e-8) & (true_norm > 1e-8)
    if np.any(valid):
        cos_sim = np.sum(v_est[valid] * true_vel[valid], axis=1) / (
            est_norm[valid] * true_norm[valid]
        )
        vel_cos_acc = float(np.mean(cos_sim > 0.0))
    else:
        vel_cos_acc = 0.0

    # Spatially Variable Gene recall & precision
    pred_svg = np.zeros(ds.n_genes, dtype=bool)
    for gstat in sp.gene_stats:
        if gstat.is_spatially_variable:
            pred_svg[gstat.gene_index] = True
    tp = int(np.sum(pred_svg & true_svg))
    fp = int(np.sum(pred_svg & ~true_svg))
    fn = int(np.sum(~pred_svg & true_svg))
    svg_recall = float(tp / max(tp + fn, 1))
    svg_precision = float(tp / max(tp + fp, 1))

    # Mean maximum fate probability (commitment sharpness on mature clusters)
    mature_mask = np.isin(true_clust, [1, 2, 3])
    fate_purity = (
        float(np.mean(np.max(pt.fate_probabilities[mature_mask], axis=1)))
        if np.any(mature_mask)
        else 0.0
    )

    # OAT sensitivity across resolution & kNN neighborhood size
    sens_deltas: Dict[str, float] = {}
    for res_alt in (0.5, 1.1):
        c_alt = cluster_dataset(ds, resolution=res_alt, method="leiden", seed=spec.seed)
        sens_deltas[f"ari_res_{res_alt:.1f}"] = adjusted_rand_index(true_clust, c_alt.labels) - ari
    # Restore main cluster labels
    ds.obs["cluster"] = clust.labels

    report = BenchmarkReport(
        ari=ari,
        nmi=nmi,
        pseudotime_spearman=pt_spearman,
        velocity_cosine_accuracy=vel_cos_acc,
        svg_recall=svg_recall,
        svg_precision=svg_precision,
        fate_purity=fate_purity,
        sensitivity_deltas=sens_deltas,
    )
    return ds, report
