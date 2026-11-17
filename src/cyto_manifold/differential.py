"""Differential expression, Wilcoxon rank-sum, GSEA, Fisher ORA, and trajectory drivers.

Provides statistical inference for cluster marker genes and pathway programs:

1. Exact tie-corrected Mann-Whitney U / Wilcoxon rank-sum test and Welch's
   t-test with log2 fold-change and Benjamini-Hochberg FDR correction.
2. Gene Set Enrichment Analysis (GSEA) using weighted Kolmogorov-Smirnov
   running enrichment score (Subramanian et al., 2005) and permutation-based
   Normalized Enrichment Score (NES) + leading-edge subset extraction.
3. Exact Fisher hypergeometric Over-Representation Analysis (ORA).
4. Pseudotime / fate-probability driver gene correlation via Spearman rank
   coefficients and cubic polynomial trend R^2.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import fisher_exact, norm, rankdata, spearmanr, t as student_t

from cyto_manifold.container import TranscriptomeDataset
from cyto_manifold.spatial import benjamini_hochberg
from cyto_manifold.types import DEResult, EnrichmentResult, FloatArray, IntArray

CANONICAL_PATHWAYS: Dict[str, Tuple[str, ...]] = {
    "STEMNESS_PLURIPOTENCY": ("SOX2", "NANOG", "OCT4", "LGR5", "MYC", "RSPO3", "LEF1", "TCF7"),
    "EMT_MESENCHYMAL": ("SNAI1", "ZEB1", "TWIST1", "FN1", "COL1A1", "PDGFA", "PDGFRA", "VIM"),
    "EPITHELIAL_DIFFERENTIATION": ("CDH1", "EPCAM", "KRT18", "KRT19", "EGF", "EGFR", "AREG", "GATA3"),
    "ANGIOGENESIS_HYPOXIA": ("VEGFA", "VEGFB", "KDR", "FLT1", "HIF1A", "FGF2", "FGFR1", "ETS1"),
    "NOTCH_LINEAGE_COMMITMENT": ("DLL1", "JAG1", "NOTCH1", "NOTCH2", "HES1", "BMP4", "BMPR1A"),
    "IMMUNE_INFLAMMATORY": ("CXCL12", "CXCR4", "CCL2", "CCR2", "IL6", "IL6R", "TNF", "NFKB1", "CSF1R"),
    "CELL_CYCLE_PROLIFERATION": ("E2F1", "MKI67", "CDK1", "PCNA", "TOP2A", "CCNB1"),
    "TGFB_FIBROSIS": ("TGFB1", "TGFBR1", "SMAD3", "COL1A1", "FN1", "SERPINE1"),
}


def wilcoxon_rank_sum_gene(x_in: FloatArray, x_out: FloatArray) -> Tuple[float, float]:
    """Compute tie-corrected two-sided Wilcoxon rank-sum z-statistic and p-value."""
    n1 = len(x_in)
    n2 = len(x_out)
    if n1 == 0 or n2 == 0:
        return 0.0, 1.0
    combined = np.concatenate([x_in, x_out])
    n_tot = n1 + n2
    ranks = rankdata(combined, method="average")
    r1 = float(np.sum(ranks[:n1]))
    u1 = r1 - 0.5 * n1 * (n1 + 1)
    mu_u = 0.5 * n1 * n2

    # Tie correction for variance
    _, tie_counts = np.unique(combined, return_counts=True)
    tie_term = float(np.sum(tie_counts ** 3 - tie_counts))
    var_u = (n1 * n2 / 12.0) * ((n_tot + 1.0) - tie_term / max(n_tot * (n_tot - 1.0), 1.0))
    if var_u <= 1e-12:
        return 0.0, 1.0
    z_val = (u1 - mu_u) / float(np.sqrt(var_u))
    p_val = float(2.0 * norm.sf(abs(z_val)))
    return float(z_val), p_val


def welch_t_test_gene(x_in: FloatArray, x_out: FloatArray) -> Tuple[float, float]:
    """Compute two-sided Welch's t-statistic and p-value."""
    n1 = len(x_in)
    n2 = len(x_out)
    if n1 < 2 or n2 < 2:
        return 0.0, 1.0
    m1 = float(np.mean(x_in))
    m2 = float(np.mean(x_out))
    v1 = float(np.var(x_in, ddof=1))
    v2 = float(np.var(x_out, ddof=1))
    se2 = v1 / n1 + v2 / n2
    if se2 <= 1e-12:
        return 0.0, 1.0
    t_stat = (m1 - m2) / float(np.sqrt(se2))
    df_num = se2 ** 2
    df_den = ((v1 / n1) ** 2) / (n1 - 1) + ((v2 / n2) ** 2) / (n2 - 1)
    df = max(df_num / max(df_den, 1e-12), 1.0)
    p_val = float(2.0 * student_t.sf(abs(t_stat), df=df))
    return float(t_stat), p_val


def rank_genes_groups(
    dataset: TranscriptomeDataset,
    labels: Optional[IntArray] = None,
    method: str = "wilcoxon",
    fdr_threshold: float = 0.05,
    min_log2_fc: float = 0.25,
) -> List[DEResult]:
    """Perform one-vs-rest differential expression across all clusters."""
    if method not in ("wilcoxon", "t-test"):
        raise ValueError(f"Unsupported DE method '{method}'; expected 'wilcoxon' or 't-test'")

    if labels is None:
        if "cluster" not in dataset.obs:
            raise ValueError("Cluster labels required in obs['cluster'] or labels argument")
        labels = np.asarray(dataset.obs["cluster"], dtype=np.int64)

    expr = dataset.X
    unique_c = sorted(int(c) for c in np.unique(labels))
    if len(unique_c) < 2:
        return []

    raw_rows: List[Tuple[int, str, float, float, float, float, float, float]] = []
    p_vals: List[float] = []
    eps = 1e-3

    for cid in unique_c:
        in_mask = labels == cid
        out_mask = ~in_mask
        if not np.any(in_mask) or not np.any(out_mask):
            continue
        x_in_all = expr[in_mask]
        x_out_all = expr[out_mask]

        for j, g_name in enumerate(dataset.gene_names):
            xi = x_in_all[:, j]
            xo = x_out_all[:, j]
            m_in = float(np.mean(xi))
            m_out = float(np.mean(xo))
            # Convert log1p scale back to linear-ish pseudocount ratio for log2FC
            lin_in = float(np.mean(np.expm1(np.maximum(xi, 0.0)))) + eps
            lin_out = float(np.mean(np.expm1(np.maximum(xo, 0.0)))) + eps
            log2_fc = float(np.log2(lin_in / lin_out))
            pct_in = float(np.mean(xi > 0.0))

            if method == "wilcoxon":
                stat, p_val = wilcoxon_rank_sum_gene(xi, xo)
            else:
                stat, p_val = welch_t_test_gene(xi, xo)

            raw_rows.append((cid, g_name, log2_fc, stat, p_val, m_in, m_out, pct_in))
            p_vals.append(p_val)

    fdrs = benjamini_hochberg(p_vals)
    results: List[DEResult] = []
    for (cid, g_name, lfc, stat, p_val, m_in, m_out, pct_in), fdr in zip(raw_rows, fdrs):
        if fdr <= fdr_threshold and abs(lfc) >= min_log2_fc:
            results.append(
                DEResult(
                    cluster=cid,
                    gene_name=g_name,
                    log2_fc=lfc,
                    statistic=stat,
                    p_value=p_val,
                    fdr=float(fdr),
                    mean_in_cluster=m_in,
                    mean_rest=m_out,
                    pct_in_cluster=pct_in,
                )
            )

    results.sort(key=lambda r: (r.cluster, r.fdr, -abs(r.log2_fc)))
    return results


def _weighted_ks_es(
    ranked_genes: Sequence[str],
    ranked_scores: FloatArray,
    gene_set: set[str],
    weight_p: float = 1.0,
) -> Tuple[float, List[str]]:
    """Compute weighted Kolmogorov-Smirnov GSEA Enrichment Score and leading edge."""
    n = len(ranked_genes)
    in_set = np.array([g in gene_set for g in ranked_genes], dtype=bool)
    n_hit = int(np.sum(in_set))
    n_miss = n - n_hit
    if n_hit == 0 or n_miss == 0:
        return 0.0, []

    weights = np.power(np.abs(ranked_scores[in_set]), weight_p)
    n_r = max(float(np.sum(weights)), 1e-12)
    step_hit = np.zeros(n, dtype=np.float64)
    step_hit[in_set] = np.power(np.abs(ranked_scores[in_set]), weight_p) / n_r
    step_miss = np.where(~in_set, 1.0 / float(n_miss), 0.0)
    running = np.cumsum(step_hit - step_miss)

    max_pos = float(np.max(running))
    min_neg = float(np.min(running))
    if abs(max_pos) >= abs(min_neg):
        peak_idx = int(np.argmax(running))
        es = max_pos
        leading = [ranked_genes[i] for i in range(peak_idx + 1) if in_set[i]]
    else:
        peak_idx = int(np.argmin(running))
        es = min_neg
        leading = [ranked_genes[i] for i in range(peak_idx, n) if in_set[i]]
    return float(es), leading


def run_gsea_enrichment(
    dataset: TranscriptomeDataset,
    labels: Optional[IntArray] = None,
    pathways: Optional[Dict[str, Tuple[str, ...]]] = None,
    n_permutations: int = 80,
    seed: int = 42,
) -> List[EnrichmentResult]:
    """Run GSEA + Fisher ORA across clusters against biological pathway signatures."""
    if labels is None:
        labels = (
            np.asarray(dataset.obs["cluster"], dtype=np.int64)
            if "cluster" in dataset.obs
            else np.zeros(dataset.n_cells, dtype=np.int64)
        )
    pw_map = pathways or CANONICAL_PATHWAYS
    expr = dataset.X
    gene_names = dataset.gene_names
    n_genes = len(gene_names)
    rng = np.random.default_rng(seed)

    results: List[EnrichmentResult] = []
    unique_c = sorted(int(c) for c in np.unique(labels))
    if len(unique_c) < 2:
        return []

    for cid in unique_c:
        in_mask = labels == cid
        out_mask = ~in_mask
        if not np.any(in_mask) or not np.any(out_mask):
            continue
        scores = np.array(
            [wilcoxon_rank_sum_gene(expr[in_mask, j], expr[out_mask, j])[0] for j in range(n_genes)],
            dtype=np.float64,
        )
        order = np.argsort(-scores, kind="mergesort")
        ranked_genes = [gene_names[int(i)] for i in order]
        ranked_scores = scores[order]
        top_up_set = set(ranked_genes[: max(5, n_genes // 5)])

        for pw_name, pw_genes in pw_map.items():
            gset = set(pw_genes) & set(gene_names)
            if len(gset) < 2:
                continue
            es, leading = _weighted_ks_es(ranked_genes, ranked_scores, gset)

            # Permutation null for NES & p-value
            null_es = np.zeros(n_permutations, dtype=np.float64)
            for b in range(n_permutations):
                perm_genes = rng.permutation(ranked_genes).tolist()
                null_es[b], _ = _weighted_ks_es(perm_genes, ranked_scores, gset)

            if es >= 0.0:
                pos_null = null_es[null_es >= 0.0]
                mean_pos = float(np.mean(pos_null)) if len(pos_null) > 0 else 1.0
                nes = es / max(mean_pos, 1e-6)
                p_val = float(np.sum(null_es >= es) + 1) / float(n_permutations + 1)
            else:
                neg_null = null_es[null_es < 0.0]
                mean_neg = float(np.mean(np.abs(neg_null))) if len(neg_null) > 0 else 1.0
                nes = es / max(mean_neg, 1e-6)
                p_val = float(np.sum(null_es <= es) + 1) / float(n_permutations + 1)

            # Fisher exact ORA against top upregulated genes
            a = len(top_up_set & gset)
            b_cnt = len(top_up_set - gset)
            c_cnt = len(gset - top_up_set)
            d_cnt = max(0, n_genes - a - b_cnt - c_cnt)
            _, ora_p = fisher_exact([[a, b_cnt], [c_cnt, d_cnt]], alternative="greater")

            results.append(
                EnrichmentResult(
                    pathway_name=pw_name,
                    cluster=cid,
                    enrichment_score=es,
                    nes=float(nes),
                    p_value=p_val,
                    ora_p_value=float(ora_p),
                    leading_edge_genes=leading,
                )
            )

    results.sort(key=lambda r: (-abs(r.nes), r.p_value))
    return results


def detect_trajectory_drivers(
    dataset: TranscriptomeDataset,
    pseudotime: FloatArray,
    top_k: int = 15,
) -> List[Tuple[str, float, float]]:
    """Identify genes strongly correlated with pseudotime via Spearman & cubic trend R^2."""
    expr = dataset.X
    drivers: List[Tuple[str, float, float]] = []
    t_norm = np.asarray(pseudotime, dtype=np.float64)
    for j, g_name in enumerate(dataset.gene_names):
        y = expr[:, j]
        if float(np.std(y)) < 1e-12:
            continue
        rho, _ = spearmanr(t_norm, y)
        rho_val = 0.0 if not np.isfinite(rho) else float(rho)
        coeffs = np.polyfit(t_norm, y, deg=min(3, max(1, len(t_norm) - 1)))
        y_hat = np.polyval(coeffs, t_norm)
        ss_res = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = float(np.clip(1.0 - ss_res / max(ss_tot, 1e-12), 0.0, 1.0))
        drivers.append((g_name, rho_val, r2))

    drivers.sort(key=lambda d: (-abs(d[1]), -d[2]))
    return drivers[:top_k]
