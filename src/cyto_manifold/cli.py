"""Command-line interface for CytoManifold.

Supports six operational subcommands runnable via either ``cyto-manifold``
or ``python -m cyto_manifold.cli``:

- ``demo``: Run full synthetic single-cell + spatial multi-modal pipeline.
- ``simulate``: Generate synthetic multi-lineage spatial tissue and save ``.npz``.
- ``analyze``: Run QC, HVGs, PCA, Harmony, UMAP, Leiden clustering, and DE/GSEA.
- ``velocity``: Run steady-state or dynamical EM RNA velocity, DPT, and fates.
- ``spatial``: Run Delaunay/kNN Moran's I SVGs, LISA hotspots, and L-R signaling.
- ``benchmark``: Run ground-truth recovery audit and OAT sensitivity analysis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import numpy as np

from cyto_manifold.benchmark import generate_synthetic_tissue, run_benchmark_audit
from cyto_manifold.clustering import cluster_dataset
from cyto_manifold.container import TranscriptomeDataset
from cyto_manifold.differential import rank_genes_groups, run_gsea_enrichment
from cyto_manifold.manifold import embed_manifold, magic_diffusion_impute
from cyto_manifold.pipeline import CytoManifoldPipeline, export_pipeline_telemetry
from cyto_manifold.preprocessing import (
    compute_pca,
    compute_qc_metrics,
    filter_and_normalize,
    integrate_batches_harmony,
    select_highly_variable_genes,
)
from cyto_manifold.pseudotime import infer_trajectory_and_fates
from cyto_manifold.spatial import analyze_spatial_tissue
from cyto_manifold.types import TissueBenchmarkSpec
from cyto_manifold.velocity import estimate_rna_velocity
from cyto_manifold.visualizer import (
    assert_ascii,
    render_ascii_scatter,
    render_executive_dashboard,
    render_phase_portrait,
    render_signaling_table,
    render_velocity_field,
)


def _load_or_simulate(
    input_path: Optional[str],
    cells: int,
    genes: int,
    batches: int,
    seed: int,
) -> TranscriptomeDataset:
    if input_path:
        return TranscriptomeDataset.load_npz(Path(input_path))
    return generate_synthetic_tissue(
        TissueBenchmarkSpec(
            n_cells=cells,
            n_genes=genes,
            n_batches=batches,
            seed=seed,
        )
    )


def _cmd_demo(args: argparse.Namespace) -> int:
    pipe = CytoManifoldPipeline(
        leiden_resolution=args.resolution,
        velocity_mode=args.velocity_mode,
        spatial_permutations=args.permutations,
        seed=args.seed,
    )
    outputs = pipe.run_synthetic_demo(
        n_cells=args.cells,
        n_genes=args.genes,
        n_batches=args.batches,
    )
    print(
        render_executive_dashboard(
            outputs.report,
            outputs.dataset,
            velocity=outputs.velocity,
            pseudotime=outputs.pseudotime,
            spatial=outputs.spatial,
            grn=outputs.grn,
        )
    )
    if args.export_dir:
        paths = export_pipeline_telemetry(outputs, Path(args.export_dir))
        print(assert_ascii(f"\nExported telemetry: {paths['report_json']}, {paths['cells_csv']}"))
    return 0


def _cmd_simulate(args: argparse.Namespace) -> int:
    spec = TissueBenchmarkSpec(
        n_cells=args.cells,
        n_genes=args.genes,
        n_batches=args.batches,
        noise_overdispersion=args.overdispersion,
        seed=args.seed,
    )
    ds = generate_synthetic_tissue(spec)
    print(
        assert_ascii(
            f"Simulated tissue: {ds.n_cells} cells x {ds.n_genes} genes | "
            f"batches={spec.n_batches} | sha256={ds.content_checksum()[:16]}"
        )
    )
    if args.output:
        saved = ds.save_npz(Path(args.output))
        print(assert_ascii(f"Saved archive: {saved}"))
    return 0


def _cmd_analyze(args: argparse.Namespace) -> int:
    raw = _load_or_simulate(args.input, args.cells, args.genes, args.batches, args.seed)
    qc = compute_qc_metrics(raw, min_genes_per_cell=5, min_cells_per_gene=2, max_doublet_score=0.98)
    ds = filter_and_normalize(raw, qc=qc)
    hvg = select_highly_variable_genes(ds, n_top_genes=min(40, ds.n_genes))
    pca = compute_pca(ds, n_components=min(18, ds.n_cells - 1, ds.n_genes))
    if "batch" in ds.obs and len(set(ds.obs["batch"].tolist())) > 1:
        pca = integrate_batches_harmony(pca, ds.obs["batch"].tolist(), seed=args.seed)
        ds.set_obsm("X_pca", pca.coordinates)
    embed_manifold(ds, n_neighbors=14, seed=args.seed)
    clust = cluster_dataset(ds, resolution=args.resolution, method="leiden", seed=args.seed)
    de = rank_genes_groups(ds, labels=clust.labels)
    gsea = run_gsea_enrichment(ds, labels=clust.labels, n_permutations=30, seed=args.seed)

    print(
        assert_ascii(
            f"Analysis complete: {ds.n_cells} cells, {int(np.sum(hvg.highly_variable_mask))} HVGs, "
            f"{clust.n_clusters} Leiden clusters (Q={clust.modularity:.3f}, Sil={clust.silhouette:.3f})"
        )
    )
    print(render_ascii_scatter(ds.obsm["X_umap"], clust.labels, title="Leiden UMAP"))
    if de:
        top_de = ", ".join(f"C{d.cluster}:{d.gene_name}(log2FC={d.log2_fc:.2f})" for d in de[:6])
        print(assert_ascii(f"Top DE markers: {top_de}"))
    if gsea:
        top_pw = ", ".join(f"C{g.cluster}:{g.pathway_name}(NES={g.nes:.2f})" for g in gsea[:4])
        print(assert_ascii(f"Top GSEA pathways: {top_pw}"))
    return 0


def _cmd_velocity(args: argparse.Namespace) -> int:
    raw = _load_or_simulate(args.input, args.cells, args.genes, 1, args.seed)
    qc = compute_qc_metrics(raw, min_genes_per_cell=5, min_cells_per_gene=2, max_doublet_score=0.98)
    ds = filter_and_normalize(raw, qc=qc)
    select_highly_variable_genes(ds, n_top_genes=min(40, ds.n_genes))
    compute_pca(ds, n_components=min(16, ds.n_cells - 1, ds.n_genes))
    graph, _ = embed_manifold(ds, n_neighbors=14, seed=args.seed)
    magic_diffusion_impute(ds, graph, t_steps=3)
    cluster_dataset(ds, resolution=0.8, seed=args.seed)
    vel = estimate_rna_velocity(ds, mode=args.mode, use_imputed=True)
    pt = infer_trajectory_and_fates(ds)

    print(
        assert_ascii(
            f"Velocity ({vel.mode}): mean R2={vel.mean_r_squared:.3f}, "
            f"mean coherence={vel.mean_confidence:.3f}, terminal clusters={pt.terminal_clusters}"
        )
    )
    print(render_velocity_field(ds.obsm["X_umap"], vel.embedding_velocity))
    target_gene = (
        args.gene
        if args.gene and args.gene in ds.gene_names
        else max(vel.gene_fits, key=lambda f: f.r_squared).gene_name
    )
    print(render_phase_portrait(ds, vel, target_gene))
    return 0


def _cmd_spatial(args: argparse.Namespace) -> int:
    raw = _load_or_simulate(args.input, args.cells, args.genes, 1, args.seed)
    qc = compute_qc_metrics(raw, min_genes_per_cell=5, min_cells_per_gene=2, max_doublet_score=0.98)
    ds = filter_and_normalize(raw, qc=qc)
    select_highly_variable_genes(ds, n_top_genes=min(40, ds.n_genes))
    compute_pca(ds, n_components=min(16, ds.n_cells - 1, ds.n_genes))
    embed_manifold(ds, n_neighbors=14, seed=args.seed)
    clust = cluster_dataset(ds, resolution=0.8, seed=args.seed)
    sp = analyze_spatial_tissue(
        ds,
        mode=args.graph_mode,
        n_permutations=args.permutations,
        seed=args.seed,
    )
    print(render_ascii_scatter(ds.obsm["spatial"], clust.labels, title="Spatial Tissue Clusters"))
    print(render_signaling_table(sp))
    return 0


def _cmd_benchmark(args: argparse.Namespace) -> int:
    spec = TissueBenchmarkSpec(
        n_cells=args.cells,
        n_genes=args.genes,
        n_batches=args.batches,
        seed=args.seed,
    )
    _, report = run_benchmark_audit(spec)
    payload = {
        "ari": round(report.ari, 4),
        "nmi": round(report.nmi, 4),
        "pseudotime_spearman": round(report.pseudotime_spearman, 4),
        "velocity_cosine_accuracy": round(report.velocity_cosine_accuracy, 4),
        "svg_recall": round(report.svg_recall, 4),
        "svg_precision": round(report.svg_precision, 4),
        "fate_purity": round(report.fate_purity, 4),
        "sensitivity_deltas": {k: round(v, 4) for k, v in report.sensitivity_deltas.items()},
    }
    print(assert_ascii("=== CytoManifold Ground-Truth Benchmark Report ==="))
    print(assert_ascii(json.dumps(payload, indent=2)))
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Create the CytoManifold multi-subcommand argument parser."""
    parser = argparse.ArgumentParser(
        prog="cyto-manifold",
        description="Single-cell and spatial transcriptomics manifold, RNA velocity, and signaling CLI.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_demo = sub.add_parser("demo", help="Run full synthetic single-cell + spatial pipeline")
    p_demo.add_argument("--cells", type=int, default=200)
    p_demo.add_argument("--genes", type=int, default=65)
    p_demo.add_argument("--batches", type=int, default=2)
    p_demo.add_argument("--resolution", type=float, default=0.8)
    p_demo.add_argument("--velocity-mode", choices=["dynamical", "steady_state"], default="dynamical")
    p_demo.add_argument("--permutations", type=int, default=500)
    p_demo.add_argument("--export-dir", type=str, default=None)
    p_demo.add_argument("--seed", type=int, default=42)
    p_demo.set_defaults(func=_cmd_demo)

    p_sim = sub.add_parser("simulate", help="Generate synthetic multi-lineage spatial tissue")
    p_sim.add_argument("--cells", type=int, default=200)
    p_sim.add_argument("--genes", type=int, default=65)
    p_sim.add_argument("--batches", type=int, default=2)
    p_sim.add_argument("--overdispersion", type=float, default=0.25)
    p_sim.add_argument("--output", type=str, default=None)
    p_sim.add_argument("--seed", type=int, default=42)
    p_sim.set_defaults(func=_cmd_simulate)

    p_ana = sub.add_parser("analyze", help="Run QC, HVGs, PCA, Harmony, Leiden, and DE/GSEA")
    p_ana.add_argument("--input", type=str, default=None)
    p_ana.add_argument("--cells", type=int, default=180)
    p_ana.add_argument("--genes", type=int, default=60)
    p_ana.add_argument("--batches", type=int, default=2)
    p_ana.add_argument("--resolution", type=float, default=0.8)
    p_ana.add_argument("--seed", type=int, default=42)
    p_ana.set_defaults(func=_cmd_analyze)

    p_vel = sub.add_parser("velocity", help="Estimate RNA velocity and absorbing lineage fates")
    p_vel.add_argument("--input", type=str, default=None)
    p_vel.add_argument("--cells", type=int, default=180)
    p_vel.add_argument("--genes", type=int, default=55)
    p_vel.add_argument("--mode", choices=["dynamical", "steady_state"], default="dynamical")
    p_vel.add_argument("--gene", type=str, default=None)
    p_vel.add_argument("--seed", type=int, default=42)
    p_vel.set_defaults(func=_cmd_velocity)

    p_sp = sub.add_parser("spatial", help="Run Moran's I SVGs, LISA, and Ligand-Receptor signaling")
    p_sp.add_argument("--input", type=str, default=None)
    p_sp.add_argument("--cells", type=int, default=180)
    p_sp.add_argument("--genes", type=int, default=60)
    p_sp.add_argument("--graph-mode", choices=["delaunay", "knn", "radius"], default="delaunay")
    p_sp.add_argument("--permutations", type=int, default=500)
    p_sp.add_argument("--seed", type=int, default=42)
    p_sp.set_defaults(func=_cmd_spatial)

    p_bench = sub.add_parser("benchmark", help="Run ground-truth recovery benchmark and sensitivity")
    p_bench.add_argument("--cells", type=int, default=200)
    p_bench.add_argument("--genes", type=int, default=65)
    p_bench.add_argument("--batches", type=int, default=2)
    p_bench.add_argument("--seed", type=int, default=42)
    p_bench.set_defaults(func=_cmd_benchmark)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entrypoint returning exit status code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
