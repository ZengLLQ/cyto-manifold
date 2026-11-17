"""Tests for analytical RNA velocity splicing ODEs, EM dynamical fitting, and flow projection."""

from __future__ import annotations

import numpy as np

from cyto_manifold.container import TranscriptomeDataset
from cyto_manifold.manifold import build_knn_graph
from cyto_manifold.velocity import (
    analytical_splicing_trajectory,
    compute_velocity_transition_matrix,
    estimate_rna_velocity,
)


def test_analytical_splicing_ode_conservation_and_continuity() -> None:
    t = np.linspace(0.0, 6.0, 601)
    alpha, beta, gamma, t_sw = 18.0, 1.2, 0.7, 2.5
    u, s, states = analytical_splicing_trajectory(t, alpha, beta, gamma, t_sw)

    # Initial condition (0, 0) and strict non-negativity
    assert np.isclose(u[0], 0.0)
    assert np.isclose(s[0], 0.0)
    assert np.all(u >= 0.0)
    assert np.all(s >= 0.0)

    # Verify numerical finite-difference derivatives match exact ODE du/dt & ds/dt
    dt = t[1] - t[0]
    du_num = np.gradient(u, dt)
    ds_num = np.gradient(s, dt)

    # Interior of induction phase (away from switch kink)
    ind_idx = (t > 0.2) & (t < t_sw - 0.2)
    assert np.allclose(du_num[ind_idx], alpha - beta * u[ind_idx], atol=1e-2)
    assert np.allclose(ds_num[ind_idx], beta * u[ind_idx] - gamma * s[ind_idx], atol=1e-2)

    # Interior of repression phase (alpha = 0)
    rep_idx = t > t_sw + 0.2
    assert np.allclose(du_num[rep_idx], -beta * u[rep_idx], atol=1e-2)
    assert np.allclose(ds_num[rep_idx], beta * u[rep_idx] - gamma * s[rep_idx], atol=1e-2)
    assert np.all(states[ind_idx] == 1.0)
    assert np.all(states[rep_idx] == -1.0)


def test_dynamical_and_steady_state_velocity_recover_latent_order() -> None:
    rng = np.random.default_rng(33)
    n_cells = 80
    t_true = np.linspace(0.1, 5.5, n_cells)
    n_genes = 8
    u_mat = np.zeros((n_cells, n_genes), dtype=np.float64)
    s_mat = np.zeros((n_cells, n_genes), dtype=np.float64)

    for g in range(n_genes):
        alpha = 15.0 + 2.0 * g
        beta = 1.0 + 0.1 * (g % 3)
        gamma = 0.6 + 0.08 * (g % 4)
        u_g, s_g, _ = analytical_splicing_trajectory(t_true, alpha, beta, gamma, t_switch=2.8)
        u_mat[:, g] = np.maximum(u_g + rng.normal(0.0, 0.25, size=n_cells), 0.0)
        s_mat[:, g] = np.maximum(s_g + rng.normal(0.0, 0.25, size=n_cells), 0.0)

    ds = TranscriptomeDataset(X=s_mat + u_mat, spliced=s_mat, unspliced=u_mat)
    graph = build_knn_graph(s_mat, n_neighbors=8)
    ds.set_obsp("connectivities", graph.connectivities)
    ds.set_obsm("X_umap", np.column_stack([t_true, np.zeros_like(t_true)]))

    res_ss = estimate_rna_velocity(ds, mode="steady_state", use_imputed=False)
    res_dyn = estimate_rna_velocity(ds, mode="dynamical", use_imputed=False)

    assert res_ss.velocities.shape == (n_cells, n_genes)
    assert res_dyn.velocities.shape == (n_cells, n_genes)
    assert res_dyn.mean_r_squared > 0.70
    assert res_dyn.mean_confidence > 0.25
    # Transition matrix rows must strictly sum to 1.0
    assert np.allclose(np.sum(res_dyn.transition_matrix, axis=1), 1.0, atol=1e-9)
    # Latent time correlates positively with true time
    corr = float(np.corrcoef(t_true, res_dyn.latent_time)[0, 1])
    assert corr > 0.75


def test_velocity_transition_matrix_row_stochastic() -> None:
    s = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    v = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 0.0]])
    conn = np.array(
        [
            [0.0, 1.0, 0.5, 0.0],
            [1.0, 0.0, 1.0, 0.5],
            [0.5, 1.0, 0.0, 1.0],
            [0.0, 0.5, 1.0, 0.0],
        ]
    )
    t_mat = compute_velocity_transition_matrix(s, v, conn, sigma_corr=0.05)
    assert np.allclose(np.sum(t_mat, axis=1), 1.0, atol=1e-12)
    # Cell 0 velocity points +x, so transition to cell 1 should exceed zero and bias forward
    assert t_mat[0, 1] > 0.5
