"""Pure 7-bit ASCII terminal visualizer for single-cell and spatial manifolds.

Every rendering function emits strictly 7-bit ASCII characters (codes 0-127)
so output renders cleanly on Windows cp1252, macOS, and headless Linux CI
consoles without ``UnicodeEncodeError``.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from cyto_manifold.container import TranscriptomeDataset
from cyto_manifold.types import (
    FloatArray,
    GRNResult,
    IntArray,
    PipelineReport,
    PseudotimeResult,
    SpatialAnalysisResult,
    VelocityResult,
)

CLUSTER_MARKERS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def assert_ascii(text: str) -> str:
    """Verify every character in ``text`` is strictly 7-bit ASCII (0 <= ord <= 127)."""
    for idx, ch in enumerate(text):
        if ord(ch) > 127:
            raise ValueError(f"Non-ASCII character {ch!r} (ord={ord(ch)}) at index {idx}")
    return text


def _direction_glyph(vx: float, vy: float) -> str:
    """Map 2D velocity vector to an ASCII directional glyph."""
    mag = float(np.hypot(vx, vy))
    if mag < 1e-6:
        return "."
    ang = float(np.degrees(np.arctan2(vy, vx)))
    if -22.5 <= ang <= 22.5:
        return ">"
    if 22.5 < ang <= 67.5:
        return "/"
    if 67.5 < ang <= 112.5:
        return "^"
    if 112.5 < ang <= 157.5:
        return "\\"
    if ang > 157.5 or ang < -157.5:
        return "<"
    if -157.5 <= ang <= -112.5:
        return "/"
    if -112.5 < ang <= -67.5:
        return "v"
    return "\\"


def render_ascii_scatter(
    coords: FloatArray,
    labels: Optional[IntArray] = None,
    title: str = "Manifold Projection",
    width: int = 58,
    height: int = 18,
) -> str:
    """Render a 2D embedding or spatial tissue map as a framed 7-bit ASCII plot."""
    pts = np.asarray(coords[:, :2], dtype=np.float64)
    if labels is None:
        labels = np.zeros(pts.shape[0], dtype=np.int64)

    x_min, x_max = float(np.min(pts[:, 0])), float(np.max(pts[:, 0]))
    y_min, y_max = float(np.min(pts[:, 1])), float(np.max(pts[:, 1]))
    x_span = max(x_max - x_min, 1e-9)
    y_span = max(y_max - y_min, 1e-9)

    grid = [[" " for _ in range(width)] for _ in range(height)]
    for i in range(pts.shape[0]):
        col = int(round((pts[i, 0] - x_min) / x_span * (width - 1)))
        row = height - 1 - int(round((pts[i, 1] - y_min) / y_span * (height - 1)))
        col = max(0, min(width - 1, col))
        row = max(0, min(height - 1, row))
        cid = int(labels[i]) % len(CLUSTER_MARKERS)
        marker = CLUSTER_MARKERS[cid]
        if grid[row][col] == " ":
            grid[row][col] = marker
        elif grid[row][col] != marker:
            grid[row][col] = "*"

    lines: List[str] = [f"+-- {title} " + "-" * max(1, width - len(title) - 4) + "+"]
    for row_chars in grid:
        lines.append("|" + "".join(row_chars) + "|")
    lines.append("+" + "-" * width + "+")

    unique_c = sorted(int(c) for c in np.unique(labels))
    legend = ", ".join(
        f"{CLUSTER_MARKERS[c % len(CLUSTER_MARKERS)]}=C{c}(n={int(np.sum(labels == c))})"
        for c in unique_c[:8]
    )
    lines.append(f"Clusters: {legend} | *=overlap")
    return assert_ascii("\n".join(lines))


def render_velocity_field(
    coords: FloatArray,
    velocities_2d: FloatArray,
    title: str = "RNA Velocity Stream Grid",
    width: int = 58,
    height: int = 16,
) -> str:
    """Render coarse-grained 2D RNA velocity arrows on an ASCII coordinate grid."""
    pts = np.asarray(coords[:, :2], dtype=np.float64)
    vel = np.asarray(velocities_2d[:, :2], dtype=np.float64)

    x_min, x_max = float(np.min(pts[:, 0])), float(np.max(pts[:, 0]))
    y_min, y_max = float(np.min(pts[:, 1])), float(np.max(pts[:, 1]))
    x_span = max(x_max - x_min, 1e-9)
    y_span = max(y_max - y_min, 1e-9)

    vx_sum = np.zeros((height, width), dtype=np.float64)
    vy_sum = np.zeros((height, width), dtype=np.float64)
    counts = np.zeros((height, width), dtype=np.int64)

    for i in range(pts.shape[0]):
        col = int(round((pts[i, 0] - x_min) / x_span * (width - 1)))
        row = height - 1 - int(round((pts[i, 1] - y_min) / y_span * (height - 1)))
        col = max(0, min(width - 1, col))
        row = max(0, min(height - 1, row))
        vx_sum[row, col] += vel[i, 0]
        vy_sum[row, col] += vel[i, 1]
        counts[row, col] += 1

    lines: List[str] = [f"+-- {title} " + "-" * max(1, width - len(title) - 4) + "+"]
    for r in range(height):
        row_chars: List[str] = []
        for c in range(width):
            if counts[r, c] == 0:
                row_chars.append(" ")
            else:
                row_chars.append(_direction_glyph(vx_sum[r, c], vy_sum[r, c]))
        lines.append("|" + "".join(row_chars) + "|")
    lines.append("+" + "-" * width + "+")
    lines.append("Arrows: >, <, ^, v, /, \\ indicate local RNA velocity flow; . = steady")
    return assert_ascii("\n".join(lines))


def render_phase_portrait(
    dataset: TranscriptomeDataset,
    velocity: VelocityResult,
    gene_name: str,
    width: int = 54,
    height: int = 15,
) -> str:
    """Render unspliced vs. spliced phase portrait with fitted dynamical curve."""
    g_idx = dataset.gene_index(gene_name)
    s_obs = dataset.layers.get("Ms", dataset.layers["spliced"])[:, g_idx]
    u_obs = dataset.layers.get("Mu", dataset.layers["unspliced"])[:, g_idx]
    s_hat = velocity.spliced_hat[:, g_idx]
    u_hat = velocity.unspliced_hat[:, g_idx]
    fit = velocity.gene_fits[g_idx]

    s_max = max(float(np.max(s_obs)), float(np.max(s_hat)), 1e-6)
    u_max = max(float(np.max(u_obs)), float(np.max(u_hat)), 1e-6)

    grid = [[" " for _ in range(width)] for _ in range(height)]
    for i in range(len(s_obs)):
        c = max(0, min(width - 1, int(round((s_obs[i] / s_max) * (width - 1)))))
        r = max(0, min(height - 1, height - 1 - int(round((u_obs[i] / u_max) * (height - 1)))))
        grid[r][c] = "."

    for i in range(len(s_hat)):
        c = max(0, min(width - 1, int(round((s_hat[i] / s_max) * (width - 1)))))
        r = max(0, min(height - 1, height - 1 - int(round((u_hat[i] / u_max) * (height - 1)))))
        grid[r][c] = "#"

    title = f"Phase Portrait: {gene_name} (R2={fit.r_squared:.3f}, mode={fit.mode})"
    lines: List[str] = [f"+-- {title} " + "-" * max(1, width - len(title) - 4) + "+"]
    for row_chars in grid:
        lines.append("|" + "".join(row_chars) + "|")
    lines.append("+" + "-" * width + "+")
    lines.append(
        f"X: Spliced [0..{s_max:.2f}] | Y: Unspliced [0..{u_max:.2f}] | .=cell, #=ODE fit"
    )
    return assert_ascii("\n".join(lines))


def render_signaling_table(
    spatial: SpatialAnalysisResult,
    max_rows: int = 8,
) -> str:
    """Render top spatially gated ligand-receptor interactions and SVG summary."""
    lines: List[str] = [
        "=== Spatial Autocorrelation & Ligand-Receptor Signaling ===",
        f"Graph mode: {spatial.graph_mode} | Spatially Variable Genes (FDR<=0.05): {spatial.n_spatially_variable}",
        "Top SVGs: "
        + ", ".join(
            f"{g.gene_name}(I={g.morans_i:.2f},z={g.z_score_i:.1f})"
            for g in spatial.gene_stats[:5]
        ),
        "",
        "Sender -> Receiver | Ligand   | Receptor | Pathway      | Score  | FDR    | Sig",
        "-" * 79,
    ]
    for s in spatial.signaling_interactions[:max_rows]:
        sig_flag = "YES" if s.significant else "no"
        lines.append(
            f"C{s.sender_cluster:<2}    -> C{s.receiver_cluster:<2}       | "
            f"{s.ligand:<8} | {s.receptor:<8} | {s.pathway:<12} | "
            f"{s.score:6.2f} | {s.fdr:6.3f} | {sig_flag}"
        )
    if not spatial.signaling_interactions:
        lines.append("(No canonical ligand-receptor pairs found above expression threshold)")
    return assert_ascii("\n".join(lines))


def render_lineage_and_grn_summary(
    pseudotime: PseudotimeResult,
    grn: GRNResult,
    labels: IntArray,
) -> str:
    """Render per-cluster pseudotime/fate absorption bars and top GRN hub regulons."""
    lines: List[str] = ["=== Trajectory Pseudotime, Fate Absorption & GRN Hubs ==="]
    term_str = ", ".join(f"C{c}" for c in pseudotime.terminal_clusters)
    lines.append(f"Root cell: {pseudotime.root_cell} | Terminal clusters: [{term_str}]")

    for cid in sorted(int(c) for c in np.unique(labels)):
        mask = labels == cid
        mean_dpt = float(np.mean(pseudotime.dpt_pseudotime[mask]))
        mean_ent = float(np.mean(pseudotime.absorption_entropy[mask]))
        mean_fate = np.mean(pseudotime.fate_probabilities[mask], axis=0)
        bar_len = int(round(mean_dpt * 20))
        bar = "#" * bar_len + "-" * (20 - bar_len)
        fate_txt = " ".join(f"T{k}:{p:.2f}" for k, p in enumerate(mean_fate))
        lines.append(
            f"  C{cid:<2} (n={int(np.sum(mask)):3d}) | DPT [{bar}] {mean_dpt:.3f} | "
            f"Potency={mean_ent:.3f} | Fates: [{fate_txt}]"
        )

    lines.append("")
    top_hubs = sorted(grn.hub_centrality.items(), key=lambda kv: -kv[1])[:6]
    hub_str = ", ".join(f"{tf}({val:.2f})" for tf, val in top_hubs)
    lines.append(
        f"GRN Edges: {len(grn.edges)} | Feedforward Triads: {grn.n_feedforward_triads} | Top Hubs: {hub_str}"
    )
    return assert_ascii("\n".join(lines))


def render_executive_dashboard(
    report: PipelineReport,
    dataset: TranscriptomeDataset,
    velocity: Optional[VelocityResult] = None,
    pseudotime: Optional[PseudotimeResult] = None,
    spatial: Optional[SpatialAnalysisResult] = None,
    grn: Optional[GRNResult] = None,
) -> str:
    """Render a multi-panel 7-bit ASCII summary dashboard of the pipeline run."""
    sections: List[str] = [
        "=" * 78,
        "CYTOMANIFOLD :: SINGLE-CELL & SPATIAL TRANSCRIPTOMICS TELEMETRY",
        "=" * 78,
        f"Cells: {report.n_cells_retained}/{report.n_cells_initial} | "
        f"Genes: {report.n_genes_retained}/{report.n_genes_initial} (HVGs: {report.n_hvg}) | "
        f"Clusters: {report.n_clusters} (Q={report.modularity:.3f}, Sil={report.silhouette:.3f})",
        f"RNA Velocity: mean R2={report.mean_velocity_r2:.3f}, "
        f"coherence={report.mean_velocity_confidence:.3f} | "
        f"Terminal Fates: {report.n_terminal_states}",
        f"Spatial SVGs: {report.n_spatially_variable_genes} | "
        f"Sig L-R Pairs: {report.n_significant_lr_pairs} | "
        f"GRN Edges: {report.n_grn_edges} | DE Markers: {report.n_de_markers}",
        f"SHA-256 Digest: {report.checksum_sha256[:24]}...",
    ]

    labels = (
        np.asarray(dataset.obs["cluster"], dtype=np.int64)
        if "cluster" in dataset.obs
        else np.zeros(dataset.n_cells, dtype=np.int64)
    )
    if "X_umap" in dataset.obsm:
        sections.append("")
        sections.append(render_ascii_scatter(dataset.obsm["X_umap"], labels, title="UMAP Manifold"))
    if "spatial" in dataset.obsm:
        sections.append("")
        sections.append(render_ascii_scatter(dataset.obsm["spatial"], labels, title="Spatial Tissue"))
    if velocity is not None and "X_umap" in dataset.obsm:
        sections.append("")
        sections.append(render_velocity_field(dataset.obsm["X_umap"], velocity.embedding_velocity))
        best_gene = max(velocity.gene_fits, key=lambda f: f.r_squared).gene_name
        sections.append("")
        sections.append(render_phase_portrait(dataset, velocity, best_gene))
    if spatial is not None:
        sections.append("")
        sections.append(render_signaling_table(spatial))
    if pseudotime is not None and grn is not None:
        sections.append("")
        sections.append(render_lineage_and_grn_summary(pseudotime, grn, labels))

    return assert_ascii("\n".join(sections))
