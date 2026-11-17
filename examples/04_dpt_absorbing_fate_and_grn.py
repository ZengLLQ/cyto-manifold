"""Example 04: Diffusion pseudotime, absorbing Markov lineage fates, GRN, and GSEA."""

from __future__ import annotations

from cyto_manifold import (
    CytoManifoldPipeline,
    detect_trajectory_drivers,
    run_gsea_enrichment,
)
from cyto_manifold.visualizer import assert_ascii, render_lineage_and_grn_summary


def main() -> None:
    pipe = CytoManifoldPipeline(velocity_mode="dynamical", spatial_permutations=30, seed=55)
    out = pipe.run_synthetic_demo(n_cells=180, n_genes=65, n_batches=2)

    print(render_lineage_and_grn_summary(out.pseudotime, out.grn, out.cluster.labels))
    drivers = detect_trajectory_drivers(out.dataset, out.pseudotime.dpt_pseudotime, top_k=6)
    driver_str = ", ".join(f"{g}(rho={rho:+.2f},R2={r2:.2f})" for g, rho, r2 in drivers)
    print(assert_ascii(f"Top pseudotime drivers: {driver_str}"))

    gsea = run_gsea_enrichment(out.dataset, labels=out.cluster.labels, n_permutations=30, seed=55)
    for item in gsea[:5]:
        print(
            assert_ascii(
                f"  GSEA C{item.cluster} | {item.pathway_name:<26} | NES={item.nes:+.2f} | p={item.p_value:.3f}"
            )
        )


if __name__ == "__main__":
    main()
