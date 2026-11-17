"""End-to-end CytoManifold orchestration pipeline and telemetry serialization.

Ties together all single-cell and spatial transcriptomics engines into a single
reproducible workflow:

    Raw/Synthetic Tissue -> QC & Doublet Filter -> CP10K + VST/Pearson HVGs ->
    PCA + Harmony Batch Integration -> Fuzzy kNN Manifold & MAGIC Smoothing ->
    Leiden Community Clustering -> Dynamical/Steady-State RNA Velocity ->
    Diffusion Maps, DPT & Absorbing Markov Fates -> Spatial Moran's I & L-R
    Signaling -> CLR + AUCell GRN Inference -> Wilcoxon DE & GSEA Pathways ->
    Telemetry Export & Checksum Audit.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from cyto_manifold.benchmark import generate_synthetic_tissue
from cyto_manifold.clustering import cluster_dataset
from cyto_manifold.container import TranscriptomeDataset
from cyto_manifold.differential import (
    detect_trajectory_drivers,
    rank_genes_groups,
    run_gsea_enrichment,
)
from cyto_manifold.grn import infer_gene_regulatory_network
from cyto_manifold.manifold import embed_manifold, magic_diffusion_impute
from cyto_manifold.preprocessing import (
    compute_pca,
    compute_qc_metrics,
    filter_and_normalize,
    integrate_batches_harmony,
    select_highly_variable_genes,
    summarize_batch_mixing,
)
from cyto_manifold.pseudotime import infer_trajectory_and_fates
from cyto_manifold.spatial import analyze_spatial_tissue
from cyto_manifold.types import (
    ClusterResult,
    DEResult,
    EnrichmentResult,
    GRNResult,
    HVGResult,
    PCAResult,
    PipelineReport,
    PseudotimeResult,
    QCMetrics,
    SpatialAnalysisResult,
    TissueBenchmarkSpec,
    VelocityResult,
)
from cyto_manifold.velocity import estimate_rna_velocity


@dataclass
class PipelineOutputs:
    """Complete bundle of processed dataset and subsystem analysis results."""

    dataset: TranscriptomeDataset
    qc: QCMetrics
    hvg: HVGResult
    pca: PCAResult
    cluster: ClusterResult
    velocity: VelocityResult
    pseudotime: PseudotimeResult
    spatial: Optional[SpatialAnalysisResult]
    grn: GRNResult
    de_markers: List[DEResult]
    gsea: List[EnrichmentResult]
    trajectory_drivers: List[Tuple[str, float, float]]
    report: PipelineReport


class CytoManifoldPipeline:
    """Configurable single-cell and spatial multi-modal analysis pipeline."""

    def __init__(
        self,
        n_top_hvg: int = 45,
        n_pcs: int = 18,
        n_neighbors: int = 14,
        leiden_resolution: float = 0.85,
        velocity_mode: str = "dynamical",
        spatial_mode: str = "delaunay",
        spatial_permutations: int = 500,
        apply_harmony: bool = True,
        seed: int = 42,
    ) -> None:
        self.n_top_hvg = n_top_hvg
        self.n_pcs = n_pcs
        self.n_neighbors = n_neighbors
        self.leiden_resolution = leiden_resolution
        self.velocity_mode = velocity_mode
        self.spatial_mode = spatial_mode
        self.spatial_permutations = spatial_permutations
        self.apply_harmony = apply_harmony
        self.seed = seed

    def run(self, raw_dataset: TranscriptomeDataset) -> PipelineOutputs:
        """Execute every analytical stage on ``raw_dataset`` and return full outputs."""
        n_cells_init = raw_dataset.n_cells
        n_genes_init = raw_dataset.n_genes

        qc = compute_qc_metrics(
            raw_dataset,
            min_genes_per_cell=5,
            min_cells_per_gene=2,
            max_doublet_score=0.98,
            seed=self.seed,
        )
        ds = filter_and_normalize(raw_dataset, qc=qc)
        hvg = select_highly_variable_genes(
            ds, n_top_genes=min(self.n_top_hvg, ds.n_genes)
        )
        pca = compute_pca(
            ds,
            n_components=min(self.n_pcs, ds.n_cells - 1, ds.n_genes),
        )

        batch_mix_before = 1.0
        batch_mix_after = 1.0
        if self.apply_harmony and "batch" in ds.obs:
            batches = [str(b) for b in ds.obs["batch"].tolist()]
            if len(set(batches)) > 1:
                batch_mix_before = summarize_batch_mixing(pca.coordinates, batches)
                pca = integrate_batches_harmony(pca, batches, seed=self.seed)
                ds.set_obsm("X_pca", pca.coordinates)
                batch_mix_after = summarize_batch_mixing(pca.coordinates, batches)

        graph, _ = embed_manifold(
            ds,
            n_neighbors=min(self.n_neighbors, ds.n_cells - 1),
            layout="umap",
            seed=self.seed,
        )
        magic_diffusion_impute(ds, graph, t_steps=3)

        clust = cluster_dataset(
            ds,
            resolution=self.leiden_resolution,
            method="leiden",
            seed=self.seed,
        )
        vel = estimate_rna_velocity(ds, mode=self.velocity_mode, use_imputed=True)
        pt = infer_trajectory_and_fates(ds, velocity_weight=0.5)

        spatial_res: Optional[SpatialAnalysisResult] = None
        if ds.has_spatial():
            spatial_res = analyze_spatial_tissue(
                ds,
                mode=self.spatial_mode,
                n_permutations=self.spatial_permutations,
                seed=self.seed,
            )

        grn_res = infer_gene_regulatory_network(ds, clr_threshold=1.0)
        de_res = rank_genes_groups(ds, labels=clust.labels, method="wilcoxon")
        gsea_res = run_gsea_enrichment(ds, labels=clust.labels, n_permutations=40, seed=self.seed)
        drivers = detect_trajectory_drivers(ds, pt.dpt_pseudotime, top_k=12)

        n_svg = spatial_res.n_spatially_variable if spatial_res is not None else 0
        n_sig_lr = (
            sum(1 for s in spatial_res.signaling_interactions if s.significant)
            if spatial_res is not None
            else 0
        )

        warnings: List[str] = []
        if not clust.is_connected:
            warnings.append("One or more clusters are disconnected under thresholded graph.")
        if vel.mean_r_squared < 0.05:
            warnings.append("Low mean splicing R^2; check sequencing depth or diffusion steps.")

        report = PipelineReport(
            n_cells_initial=n_cells_init,
            n_genes_initial=n_genes_init,
            n_cells_retained=ds.n_cells,
            n_genes_retained=ds.n_genes,
            n_hvg=int(np.sum(hvg.highly_variable_mask)),
            n_clusters=clust.n_clusters,
            modularity=clust.modularity,
            silhouette=clust.silhouette,
            mean_velocity_r2=vel.mean_r_squared,
            mean_velocity_confidence=vel.mean_confidence,
            n_terminal_states=len(pt.terminal_clusters),
            n_spatially_variable_genes=n_svg,
            n_significant_lr_pairs=n_sig_lr,
            n_grn_edges=len(grn_res.edges),
            n_de_markers=len(de_res),
            checksum_sha256=ds.content_checksum(),
            diagnostics={
                "batch_mixing_before": batch_mix_before,
                "batch_mixing_after": batch_mix_after,
                "calinski_harabasz": clust.calinski_harabasz,
                "paga_mean_connectivity": float(np.mean(pt.paga_connectivity)),
                "grn_feedforward_triads": grn_res.n_feedforward_triads,
                "top_driver_genes": [d[0] for d in drivers[:5]],
            },
            warnings=warnings,
            notes=(
                "Pure-Python NumPy/SciPy single-cell & spatial pipeline",
                f"Velocity mode: {self.velocity_mode}",
            ),
        )

        return PipelineOutputs(
            dataset=ds,
            qc=qc,
            hvg=hvg,
            pca=pca,
            cluster=clust,
            velocity=vel,
            pseudotime=pt,
            spatial=spatial_res,
            grn=grn_res,
            de_markers=de_res,
            gsea=gsea_res,
            trajectory_drivers=drivers,
            report=report,
        )

    def run_synthetic_demo(
        self,
        n_cells: int = 200,
        n_genes: int = 65,
        n_batches: int = 2,
    ) -> PipelineOutputs:
        """Generate synthetic multi-lineage spatial tissue and execute the full pipeline."""
        spec = TissueBenchmarkSpec(
            n_cells=n_cells,
            n_genes=n_genes,
            n_batches=n_batches,
            seed=self.seed,
        )
        raw = generate_synthetic_tissue(spec)
        return self.run(raw)


def export_pipeline_telemetry(outputs: PipelineOutputs, output_dir: Path | str) -> Dict[str, Path]:
    """Export pipeline summary JSON and per-cell / spatial CSV tables."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / "pipeline_report.json"
    summary_payload: Dict[str, Any] = asdict(outputs.report)
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    cells_csv = out_dir / "cell_telemetry.csv"
    ds = outputs.dataset
    with cells_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "cell_id",
                "cluster",
                "dpt_pseudotime",
                "latent_time",
                "velocity_confidence",
                "absorption_entropy",
                "umap_1",
                "umap_2",
            ]
        )
        umap = ds.obsm["X_umap"]
        for i, cid in enumerate(ds.cell_ids):
            writer.writerow(
                [
                    cid,
                    int(outputs.cluster.labels[i]),
                    f"{outputs.pseudotime.dpt_pseudotime[i]:.6f}",
                    f"{outputs.velocity.latent_time[i]:.6f}",
                    f"{outputs.velocity.velocity_confidence[i]:.6f}",
                    f"{outputs.pseudotime.absorption_entropy[i]:.6f}",
                    f"{umap[i, 0]:.6f}",
                    f"{umap[i, 1]:.6f}",
                ]
            )

    return {"report_json": summary_path, "cells_csv": cells_csv}
