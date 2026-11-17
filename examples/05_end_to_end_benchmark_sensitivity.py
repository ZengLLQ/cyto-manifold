"""Example 05: Ground-truth recovery benchmark, sensitivity sweep, and telemetry export."""

from __future__ import annotations

from pathlib import Path
import tempfile

from cyto_manifold import CytoManifoldPipeline, TissueBenchmarkSpec, export_pipeline_telemetry, run_benchmark_audit
from cyto_manifold.visualizer import assert_ascii


def main() -> None:
    spec = TissueBenchmarkSpec(n_cells=200, n_genes=65, n_batches=2, seed=42)
    _, report = run_benchmark_audit(spec)
    print(assert_ascii("=== Ground-Truth Benchmark Recovery ==="))
    print(
        assert_ascii(
            f"ARI={report.ari:.3f} | NMI={report.nmi:.3f} | "
            f"Pseudotime Spearman={report.pseudotime_spearman:.3f} | "
            f"Velocity Cosine Acc={report.velocity_cosine_accuracy:.3f}"
        )
    )
    print(
        assert_ascii(
            f"SVG Recall={report.svg_recall:.3f} | SVG Precision={report.svg_precision:.3f} | "
            f"Fate Purity={report.fate_purity:.3f}"
        )
    )
    print(assert_ascii(f"Resolution sensitivity deltas: {report.sensitivity_deltas}"))

    pipe = CytoManifoldPipeline(seed=42)
    outputs = pipe.run_synthetic_demo(n_cells=160, n_genes=55, n_batches=2)
    with tempfile.TemporaryDirectory() as tmpdir:
        exported = export_pipeline_telemetry(outputs, Path(tmpdir))
        print(assert_ascii(f"Telemetry export verified: {list(exported.keys())}"))


if __name__ == "__main__":
    main()
