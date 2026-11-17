"""Tests for strict 7-bit ASCII visualization and all CLI subcommands."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from cyto_manifold.cli import main as cli_main
from cyto_manifold.pipeline import CytoManifoldPipeline
from cyto_manifold.visualizer import assert_ascii, render_executive_dashboard


def test_assert_ascii_rejects_non_ascii() -> None:
    assert assert_ascii("pure-ascii 123 +-|#*") == "pure-ascii 123 +-|#*"
    with pytest.raises(ValueError, match="Non-ASCII"):
        assert_ascii("bad unicode: \u2500")


def test_executive_dashboard_strict_ascii() -> None:
    pipe = CytoManifoldPipeline(spatial_permutations=20, seed=42)
    outputs = pipe.run_synthetic_demo(n_cells=120, n_genes=50, n_batches=1)
    dash = render_executive_dashboard(
        outputs.report,
        outputs.dataset,
        velocity=outputs.velocity,
        pseudotime=outputs.pseudotime,
        spatial=outputs.spatial,
        grn=outputs.grn,
    )
    assert "CYTOMANIFOLD" in dash
    assert "UMAP Manifold" in dash
    assert "Spatial Tissue" in dash
    assert "RNA Velocity Stream Grid" in dash
    assert_ascii(dash)
    dash.encode("cp1252")


def test_cli_all_subcommands(tmp_path, capsys) -> None:
    export_dir = tmp_path / "export"
    npz_path = tmp_path / "sim.npz"

    assert cli_main(["simulate", "--cells", "100", "--genes", "45", "--output", str(npz_path), "--seed", "7"]) == 0
    assert npz_path.exists()

    assert cli_main(["analyze", "--input", str(npz_path), "--seed", "7"]) == 0
    assert cli_main(["velocity", "--input", str(npz_path), "--mode", "steady_state", "--seed", "7"]) == 0
    assert cli_main(["spatial", "--input", str(npz_path), "--permutations", "20", "--seed", "7"]) == 0
    assert cli_main(["benchmark", "--cells", "120", "--genes", "50", "--seed", "7"]) == 0
    assert cli_main(["demo", "--cells", "120", "--genes", "50", "--export-dir", str(export_dir), "--seed", "7"]) == 0

    assert (export_dir / "pipeline_report.json").exists()
    out = capsys.readouterr().out
    assert_ascii(out)
    assert "CYTOMANIFOLD" in out


def test_cli_module_invocation_help() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "cyto_manifold.cli", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "cyto-manifold" in proc.stdout
    assert "benchmark" in proc.stdout
    assert json.dumps({"ok": True})
