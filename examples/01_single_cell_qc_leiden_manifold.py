"""Example 01: Single-cell QC, Pearson residuals, Harmony batch integration, and Leiden manifold."""

from __future__ import annotations

from cyto_manifold import (
    TissueBenchmarkSpec,
    adjusted_rand_index,
    cluster_dataset,
    compute_pca,
    compute_qc_metrics,
    embed_manifold,
    filter_and_normalize,
    generate_synthetic_tissue,
    integrate_batches_harmony,
    select_highly_variable_genes,
)
from cyto_manifold.preprocessing import summarize_batch_mixing
from cyto_manifold.visualizer import assert_ascii, render_ascii_scatter


def main() -> None:
    spec = TissueBenchmarkSpec(n_cells=220, n_genes=65, n_batches=3, batch_effect_strength=0.75, seed=42)
    raw = generate_synthetic_tissue(spec)
    qc = compute_qc_metrics(raw, min_genes_per_cell=5, min_cells_per_gene=2, max_doublet_score=0.98)
    ds = filter_and_normalize(raw, qc=qc)
    hvg = select_highly_variable_genes(ds, n_top_genes=40)
    pca_raw = compute_pca(ds, n_components=16)
    batches = ds.obs["batch"].tolist()
    mix_before = summarize_batch_mixing(pca_raw.coordinates, batches)

    pca_corr = integrate_batches_harmony(pca_raw, batches, seed=42)
    ds.set_obsm("X_pca", pca_corr.coordinates)
    mix_after = summarize_batch_mixing(pca_corr.coordinates, batches)

    embed_manifold(ds, n_neighbors=14, layout="umap", seed=42)
    clust = cluster_dataset(ds, resolution=0.75, method="leiden", seed=42)
    ari = adjusted_rand_index(ds.obs["true_cluster"], clust.labels)

    print(assert_ascii(f"QC retained: {ds.n_cells}/{raw.n_cells} cells, HVGs={int(hvg.highly_variable_mask.sum())}"))
    print(assert_ascii(f"Batch mixing entropy: before={mix_before:.3f} -> after Harmony={mix_after:.3f}"))
    print(
        assert_ascii(
            f"Leiden clusters={clust.n_clusters}, Q={clust.modularity:.3f}, "
            f"connected={clust.is_connected}, ARI={ari:.3f}"
        )
    )
    print(render_ascii_scatter(ds.obsm["X_umap"], clust.labels, title="Harmony-Integrated UMAP"))


if __name__ == "__main__":
    main()
