"""Core domain dataclasses and validation invariants for CytoManifold.

This module defines strongly-typed containers for single-cell and spatial
transcriptomics representations, RNA velocity splicing parameters, diffusion
pseudotime and Markov fate results, spatial autocorrelation statistics,
ligand-receptor signaling scores, gene regulatory network edges, and
end-to-end pipeline diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]


def validate_matrix_shape(
    matrix: NDArray[Any],
    n_rows: int,
    n_cols: int,
    name: str,
) -> FloatArray:
    """Validate and coerce a 2D numeric array to finite float64 shape.

    Parameters
    ----------
    matrix:
        Input array expected to match ``(n_rows, n_cols)``.
    n_rows:
        Expected number of observations (cells or spatial spots).
    n_cols:
        Expected number of features (genes).
    name:
        Human-readable matrix identifier for diagnostics.

    Returns
    -------
    FloatArray
        Validated 2D float64 array.
    """
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2-dimensional, got ndim={arr.ndim}")
    if arr.shape != (n_rows, n_cols):
        raise ValueError(
            f"{name} shape mismatch: expected ({n_rows}, {n_cols}), got {arr.shape}"
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values (NaN or Inf)")
    return arr


def validate_non_negative(arr: FloatArray, name: str) -> None:
    """Assert that every element in ``arr`` is non-negative."""
    if np.any(arr < -1e-12):
        min_val = float(np.min(arr))
        raise ValueError(f"{name} must be non-negative; minimum found: {min_val:.6g}")


@dataclass(frozen=True)
class QCMetrics:
    """Per-cell and per-gene quality control diagnostics."""

    total_counts: FloatArray
    n_genes_detected: IntArray
    unspliced_fraction: FloatArray
    mito_fraction: FloatArray
    doublet_scores: FloatArray
    cell_pass_mask: BoolArray
    gene_pass_mask: BoolArray
    n_cells_retained: int
    n_genes_retained: int


@dataclass(frozen=True)
class HVGResult:
    """Highly Variable Gene (HVG) selection metrics via VST / Pearson residuals."""

    gene_names: List[str]
    means: FloatArray
    variances: FloatArray
    expected_variances: FloatArray
    normalized_variances: FloatArray
    pearson_residual_vars: FloatArray
    highly_variable_mask: BoolArray
    top_indices: IntArray


@dataclass(frozen=True)
class PCAResult:
    """Truncated SVD principal component embedding and variance decomposition."""

    coordinates: FloatArray
    loadings: FloatArray
    singular_values: FloatArray
    explained_variance: FloatArray
    explained_variance_ratio: FloatArray
    mean_vector: FloatArray
    batch_corrected: bool = False


@dataclass(frozen=True)
class NeighborGraph:
    """Weighted cell-cell similarity and distance graphs."""

    n_cells: int
    n_neighbors: int
    metric: str
    indices: IntArray
    distances: FloatArray
    connectivities: FloatArray
    sigmas: FloatArray
    rhos: FloatArray


@dataclass(frozen=True)
class ClusterResult:
    """Community detection result and cluster quality metrics."""

    labels: IntArray
    n_clusters: int
    modularity: float
    silhouette: float
    calinski_harabasz: float
    is_connected: bool
    resolution: float
    method: str
    cluster_sizes: Dict[int, int] = field(default_factory=dict)


@dataclass(frozen=True)
class GeneVelocityFit:
    """Fitted splicing kinetic parameters for a single gene."""

    gene_name: str
    gene_index: int
    alpha: float
    beta: float
    gamma: float
    t_switch: float
    r_squared: float
    log_likelihood: float
    noise_sigma: float
    mode: str


@dataclass(frozen=True)
class VelocityResult:
    """Genome-wide RNA velocity estimates, transition matrix, and 2D projections."""

    velocities: FloatArray
    spliced_hat: FloatArray
    unspliced_hat: FloatArray
    latent_time_genes: FloatArray
    latent_time: FloatArray
    velocity_confidence: FloatArray
    transition_matrix: FloatArray
    embedding_velocity: FloatArray
    gene_fits: List[GeneVelocityFit]
    mode: str
    mean_r_squared: float
    mean_confidence: float


@dataclass(frozen=True)
class PseudotimeResult:
    """Diffusion map coordinates, DPT ordering, PAGA graph, and fate probabilities."""

    diffusion_components: FloatArray
    diffusion_eigenvalues: FloatArray
    root_cell: int
    dpt_pseudotime: FloatArray
    terminal_states: IntArray
    terminal_clusters: List[int]
    fate_probabilities: FloatArray
    absorption_entropy: FloatArray
    combined_transition: FloatArray
    paga_connectivity: FloatArray
    paga_cluster_ids: List[int]


@dataclass(frozen=True)
class SpatialGeneStat:
    """Spatial autocorrelation metrics for a single gene."""

    gene_name: str
    gene_index: int
    morans_i: float
    expected_i: float
    z_score_i: float
    p_value_i: float
    fdr_i: float
    gearys_c: float
    z_score_c: float
    is_spatially_variable: bool


@dataclass(frozen=True)
class LigandReceptorPair:
    """Canonical ligand-receptor interaction definition."""

    ligand: str
    receptor: str
    pathway: str
    interaction_type: str = "Secreted"


@dataclass(frozen=True)
class SignalingInteraction:
    """Statistically scored cell-cell spatial ligand-receptor communication."""

    sender_cluster: int
    receiver_cluster: int
    ligand: str
    receptor: str
    pathway: str
    score: float
    p_value: float
    fdr: float
    n_spatial_edges: int
    significant: bool


@dataclass(frozen=True)
class SpatialAnalysisResult:
    """Aggregate spatial transcriptomics, LISA hotspot, and signaling outputs."""

    spatial_weights: FloatArray
    gene_stats: List[SpatialGeneStat]
    n_spatially_variable: int
    local_moran: FloatArray
    niche_labels: IntArray
    signaling_interactions: List[SignalingInteraction]
    graph_mode: str


@dataclass(frozen=True)
class RegulonEdge:
    """Directed transcription factor to target gene regulatory interaction."""

    tf_name: str
    target_name: str
    clr_score: float
    mutual_information: float
    regression_weight: float
    velocity_lag_corr: float
    direction_sign: int


@dataclass(frozen=True)
class GRNResult:
    """Inferred gene regulatory network, AUCell regulon scores, and hub rankings."""

    edges: List[RegulonEdge]
    aucell_scores: FloatArray
    tf_names: List[str]
    hub_centrality: Dict[str, float]
    n_feedforward_triads: int


@dataclass(frozen=True)
class DEResult:
    """Differential expression result for one gene in a cluster comparison."""

    cluster: int
    gene_name: str
    log2_fc: float
    statistic: float
    p_value: float
    fdr: float
    mean_in_cluster: float
    mean_rest: float
    pct_in_cluster: float


@dataclass(frozen=True)
class EnrichmentResult:
    """Gene set enrichment (GSEA / Fisher ORA) score for a biological pathway."""

    pathway_name: str
    cluster: int
    enrichment_score: float
    nes: float
    p_value: float
    ora_p_value: float
    leading_edge_genes: List[str]


@dataclass(frozen=True)
class TissueBenchmarkSpec:
    """Configuration for synthetic multi-lineage spatial tissue generation."""

    n_cells: int = 240
    n_genes: int = 80
    n_batches: int = 2
    noise_overdispersion: float = 0.25
    spatial_radius: float = 1.6
    batch_effect_strength: float = 0.65
    seed: int = 42


@dataclass(frozen=True)
class BenchmarkReport:
    """Quantitative ground-truth recovery and sensitivity audit metrics."""

    ari: float
    nmi: float
    pseudotime_spearman: float
    velocity_cosine_accuracy: float
    svg_recall: float
    svg_precision: float
    fate_purity: float
    sensitivity_deltas: Dict[str, float]


@dataclass(frozen=True)
class PipelineReport:
    """Summary metrics emitted by an end-to-end CytoManifold pipeline execution."""

    n_cells_initial: int
    n_genes_initial: int
    n_cells_retained: int
    n_genes_retained: int
    n_hvg: int
    n_clusters: int
    modularity: float
    silhouette: float
    mean_velocity_r2: float
    mean_velocity_confidence: float
    n_terminal_states: int
    n_spatially_variable_genes: int
    n_significant_lr_pairs: int
    n_grn_edges: int
    n_de_markers: int
    checksum_sha256: str
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    notes: Tuple[str, ...] = field(default_factory=tuple)
