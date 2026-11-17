"""CytoManifold: Single-cell and spatial transcriptomics manifold, RNA velocity, and signaling engine."""

from cyto_manifold.benchmark import generate_synthetic_tissue, run_benchmark_audit
from cyto_manifold.clustering import (
    adjusted_rand_index,
    cluster_dataset,
    compute_modularity,
    leiden_cluster,
    louvain_cluster,
    normalized_mutual_information,
)
from cyto_manifold.container import TranscriptomeDataset
from cyto_manifold.differential import (
    detect_trajectory_drivers,
    rank_genes_groups,
    run_gsea_enrichment,
)
from cyto_manifold.grn import infer_gene_regulatory_network
from cyto_manifold.manifold import (
    build_knn_graph,
    compute_umap_layout,
    embed_manifold,
    magic_diffusion_impute,
)
from cyto_manifold.pipeline import CytoManifoldPipeline, PipelineOutputs, export_pipeline_telemetry
from cyto_manifold.preprocessing import (
    compute_pca,
    compute_pearson_residuals,
    compute_qc_metrics,
    filter_and_normalize,
    integrate_batches_harmony,
    select_highly_variable_genes,
)
from cyto_manifold.pseudotime import (
    compute_absorbing_fates,
    compute_diffusion_map,
    compute_diffusion_pseudotime,
    infer_trajectory_and_fates,
)
from cyto_manifold.spatial import (
    analyze_spatial_tissue,
    build_spatial_graph,
    compute_gearys_c,
    compute_local_moran,
    compute_morans_i,
    detect_spatially_variable_genes,
    score_ligand_receptor_signaling,
)
from cyto_manifold.types import TissueBenchmarkSpec
from cyto_manifold.velocity import analytical_splicing_trajectory, estimate_rna_velocity

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "CytoManifoldPipeline",
    "PipelineOutputs",
    "TissueBenchmarkSpec",
    "TranscriptomeDataset",
    "adjusted_rand_index",
    "analyze_spatial_tissue",
    "analytical_splicing_trajectory",
    "build_knn_graph",
    "build_spatial_graph",
    "cluster_dataset",
    "compute_absorbing_fates",
    "compute_diffusion_map",
    "compute_diffusion_pseudotime",
    "compute_gearys_c",
    "compute_local_moran",
    "compute_modularity",
    "compute_morans_i",
    "compute_pca",
    "compute_pearson_residuals",
    "compute_qc_metrics",
    "compute_umap_layout",
    "detect_spatially_variable_genes",
    "detect_trajectory_drivers",
    "embed_manifold",
    "estimate_rna_velocity",
    "export_pipeline_telemetry",
    "filter_and_normalize",
    "generate_synthetic_tissue",
    "infer_gene_regulatory_network",
    "infer_trajectory_and_fates",
    "integrate_batches_harmony",
    "leiden_cluster",
    "louvain_cluster",
    "magic_diffusion_impute",
    "normalized_mutual_information",
    "rank_genes_groups",
    "run_benchmark_audit",
    "run_gsea_enrichment",
    "score_ligand_receptor_signaling",
    "select_highly_variable_genes",
]
