"""Example 03: Spatial Delaunay graph, Moran's I / Geary's C SVGs, LISA, and L-R signaling."""

from __future__ import annotations

from cyto_manifold import (
    TissueBenchmarkSpec,
    analyze_spatial_tissue,
    cluster_dataset,
    compute_pca,
    compute_qc_metrics,
    embed_manifold,
    filter_and_normalize,
    generate_synthetic_tissue,
    select_highly_variable_genes,
)
from cyto_manifold.visualizer import assert_ascii, render_ascii_scatter, render_signaling_table


def main() -> None:
    raw = generate_synthetic_tissue(TissueBenchmarkSpec(n_cells=200, n_genes=65, n_batches=1, seed=77))
    qc = compute_qc_metrics(raw, min_genes_per_cell=5, min_cells_per_gene=2, max_doublet_score=0.98)
    ds = filter_and_normalize(raw, qc=qc)
    select_highly_variable_genes(ds, n_top_genes=45)
    compute_pca(ds, n_components=15)
    embed_manifold(ds, n_neighbors=14, seed=77)
    clust = cluster_dataset(ds, resolution=0.75, seed=77)

    sp = analyze_spatial_tissue(ds, mode="delaunay", n_permutations=500, seed=77)
    print(
        assert_ascii(
            f"Detected {sp.n_spatially_variable} spatially variable genes; "
            f"{len(set(sp.niche_labels))} spatial niches"
        )
    )
    print(render_ascii_scatter(ds.obsm["spatial"], sp.niche_labels, title="Spatial Tissue Niches"))
    print(render_ascii_scatter(ds.obsm["spatial"], clust.labels, title="Transcriptomic Clusters in Tissue"))
    print(render_signaling_table(sp, max_rows=10))


if __name__ == "__main__":
    main()
