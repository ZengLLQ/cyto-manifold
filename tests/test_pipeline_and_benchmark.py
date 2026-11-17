"""End-to-end pipeline integration, benchmark ground-truth recovery, and telemetry tests."""

from __future__ import annotations

import json

from cyto_manifold.benchmark import run_benchmark_audit
from cyto_manifold.pipeline import CytoManifoldPipeline, export_pipeline_telemetry
from cyto_manifold.types import TissueBenchmarkSpec


def test_end_to_end_pipeline_and_telemetry_export(tmp_path) -> None:
    pipe = CytoManifoldPipeline(
        leiden_resolution=0.75,
        velocity_mode="dynamical",
        spatial_permutations=30,
        seed=42,
    )
    outputs = pipe.run_synthetic_demo(n_cells=140, n_genes=55, n_batches=2)
    rep = outputs.report

    assert rep.n_cells_retained > 100
    assert rep.n_clusters >= 3
    assert rep.modularity > 0.3
    assert outputs.cluster.is_connected is True
    assert rep.mean_velocity_r2 > 0.05
    assert rep.n_terminal_states >= 1
    assert rep.n_spatially_variable_genes >= 10
    assert rep.n_grn_edges >= 1
    assert len(rep.checksum_sha256) == 64

    exported = export_pipeline_telemetry(outputs, tmp_path)
    assert exported["report_json"].exists()
    assert exported["cells_csv"].exists()

    parsed = json.loads(exported["report_json"].read_text(encoding="utf-8"))
    assert parsed["checksum_sha256"] == rep.checksum_sha256
    csv_lines = exported["cells_csv"].read_text(encoding="utf-8").strip().splitlines()
    assert len(csv_lines) == outputs.dataset.n_cells + 1


def test_benchmark_ground_truth_recovery_and_sensitivity() -> None:
    spec = TissueBenchmarkSpec(n_cells=180, n_genes=65, n_batches=2, seed=42)
    ds, report = run_benchmark_audit(spec)

    assert ds.n_cells > 140
    assert report.ari >= 0.50
    assert report.nmi >= 0.60
    assert report.pseudotime_spearman >= 0.50
    assert report.velocity_cosine_accuracy >= 0.55
    assert report.svg_recall >= 0.55
    assert report.fate_purity >= 0.50
    assert "ari_res_0.5" in report.sensitivity_deltas
    assert "ari_res_1.1" in report.sensitivity_deltas
