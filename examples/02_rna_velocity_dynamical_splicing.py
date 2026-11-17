"""Example 02: Dynamical EM RNA velocity splicing kinetics vs. steady-state regression."""

from __future__ import annotations

from cyto_manifold import (
    TissueBenchmarkSpec,
    cluster_dataset,
    compute_pca,
    compute_qc_metrics,
    embed_manifold,
    estimate_rna_velocity,
    filter_and_normalize,
    generate_synthetic_tissue,
    magic_diffusion_impute,
    select_highly_variable_genes,
)
from cyto_manifold.visualizer import assert_ascii, render_phase_portrait, render_velocity_field


def main() -> None:
    raw = generate_synthetic_tissue(TissueBenchmarkSpec(n_cells=180, n_genes=55, n_batches=1, seed=101))
    qc = compute_qc_metrics(raw, min_genes_per_cell=5, min_cells_per_gene=2, max_doublet_score=0.98)
    ds = filter_and_normalize(raw, qc=qc)
    select_highly_variable_genes(ds, n_top_genes=35)
    compute_pca(ds, n_components=15)
    graph, _ = embed_manifold(ds, n_neighbors=14, seed=101)
    magic_diffusion_impute(ds, graph, t_steps=3)
    cluster_dataset(ds, resolution=0.75, seed=101)

    vel_ss = estimate_rna_velocity(ds, mode="steady_state", use_imputed=True)
    vel_dyn = estimate_rna_velocity(ds, mode="dynamical", use_imputed=True)

    print(
        assert_ascii(
            f"Steady-state velocity: mean R2={vel_ss.mean_r_squared:.3f}, "
            f"coherence={vel_ss.mean_confidence:.3f}"
        )
    )
    print(
        assert_ascii(
            f"Dynamical EM velocity: mean R2={vel_dyn.mean_r_squared:.3f}, "
            f"coherence={vel_dyn.mean_confidence:.3f}"
        )
    )
    print(render_velocity_field(ds.obsm["X_umap"], vel_dyn.embedding_velocity))
    top_gene = max(vel_dyn.gene_fits, key=lambda f: f.r_squared).gene_name
    print(render_phase_portrait(ds, vel_dyn, top_gene))


if __name__ == "__main__":
    main()
