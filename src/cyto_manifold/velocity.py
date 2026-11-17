"""RNA velocity splicing kinetics (steady-state and full dynamical EM models).

Models unspliced pre-mRNA ``u(t)`` and spliced mature mRNA ``s(t)`` under the
linear transcriptional splicing ordinary differential equation system:

    du/dt = alpha_k(t) - beta * u(t)
    ds/dt = beta * u(t) - gamma * s(t)

Supports two estimation regimes:
1. ``steady_state`` (La Manno et al., 2018): extreme-quantile linear regression
   estimating ``gamma_tilde = gamma / beta`` and instantaneous residual
   velocity ``v_s = u - gamma_tilde * s``.
2. ``dynamical`` (Bergen et al., 2020): analytical induction/repression phase
   trajectory solving ``(u(t), s(t))`` paired with Expectation-Maximization
   (per-cell latent time & state assignment E-step + Nelder-Mead kinetic
   parameter optimization M-step) to recover ``(alpha, beta, gamma, t_switch)``
   and gene-shared latent time.

Also computes Boltzmann cosine-correlation cell transition probabilities
``pi_{ij}``, local velocity coherence confidence, and 2D embedding vector
projections.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from scipy.optimize import minimize

from cyto_manifold.container import TranscriptomeDataset
from cyto_manifold.types import FloatArray, GeneVelocityFit, VelocityResult


def analytical_splicing_trajectory(
    t: FloatArray,
    alpha: float,
    beta: float,
    gamma: float,
    t_switch: float,
) -> Tuple[FloatArray, FloatArray, FloatArray]:
    """Evaluate exact analytical ``(u(t), s(t), state)`` for induction and repression.

    During induction (``0 <= t <= t_switch``), transcription is active at rate
    ``alpha`` starting from ``(0, 0)``. During repression (``t > t_switch``),
    transcription drops to ``0`` starting from ``(u(t_switch), s(t_switch))``.
    Includes a small-epsilon guard when ``|beta - gamma| < 1e-6``.
    """
    t_arr = np.maximum(np.asarray(t, dtype=np.float64), 0.0)
    beta = max(float(beta), 1e-5)
    gamma = max(float(gamma), 1e-5)
    alpha = max(float(alpha), 1e-6)
    t_sw = max(float(t_switch), 1e-4)

    # Guard beta == gamma singularity in analytical integration
    if abs(beta - gamma) < 1e-5:
        gamma = beta + 1e-5

    u_out = np.zeros_like(t_arr)
    s_out = np.zeros_like(t_arr)
    states = np.zeros_like(t_arr)

    ind_mask = t_arr <= t_sw
    rep_mask = ~ind_mask

    # Induction phase: alpha_on = alpha, u(0)=0, s(0)=0
    if np.any(ind_mask):
        tau = t_arr[ind_mask]
        exp_b = np.exp(-beta * tau)
        exp_g = np.exp(-gamma * tau)
        u_ind = (alpha / beta) * (1.0 - exp_b)
        s_ind = (alpha / gamma) * (1.0 - exp_g) + (alpha / (gamma - beta)) * (exp_g - exp_b)
        u_out[ind_mask] = u_ind
        s_out[ind_mask] = np.maximum(s_ind, 0.0)
        states[ind_mask] = 1.0

    # Switch boundary values
    exp_b_sw = float(np.exp(-beta * t_sw))
    exp_g_sw = float(np.exp(-gamma * t_sw))
    u_sw = (alpha / beta) * (1.0 - exp_b_sw)
    s_sw = (alpha / gamma) * (1.0 - exp_g_sw) + (alpha / (gamma - beta)) * (exp_g_sw - exp_b_sw)
    s_sw = max(s_sw, 0.0)

    # Repression phase: alpha_off = 0 starting at (u_sw, s_sw)
    if np.any(rep_mask):
        tau_r = t_arr[rep_mask] - t_sw
        exp_br = np.exp(-beta * tau_r)
        exp_gr = np.exp(-gamma * tau_r)
        u_rep = u_sw * exp_br
        s_rep = s_sw * exp_gr + (beta * u_sw / (gamma - beta)) * (exp_br - exp_gr)
        u_out[rep_mask] = u_rep
        s_out[rep_mask] = np.maximum(s_rep, 0.0)
        states[rep_mask] = -1.0

    return u_out, s_out, states


def _fit_steady_state_gene(
    u: FloatArray,
    s: FloatArray,
    gene_name: str,
    gene_index: int,
    percentile: float = 5.0,
) -> Tuple[FloatArray, FloatArray, FloatArray, FloatArray, GeneVelocityFit]:
    """Fit steady-state degradation ratio gamma/beta using extreme upper quantiles."""
    s_max = float(np.max(s))
    u_max = float(np.max(u))
    if s_max < 1e-8 or u_max < 1e-8:
        zeros = np.zeros_like(s)
        fit = GeneVelocityFit(
            gene_name=gene_name,
            gene_index=gene_index,
            alpha=0.0,
            beta=1.0,
            gamma=1.0,
            t_switch=1.0,
            r_squared=0.0,
            log_likelihood=0.0,
            noise_sigma=1.0,
            mode="steady_state",
        )
        return zeros, s.copy(), u.copy(), zeros, fit

    norm_dist = (s / s_max) + (u / u_max)
    hi_thresh = float(np.percentile(norm_dist, 100.0 - percentile))
    lo_thresh = float(np.percentile(norm_dist, percentile))
    extreme = (norm_dist >= hi_thresh) | (norm_dist <= lo_thresh)

    if int(np.sum(extreme)) < 4:
        extreme = np.ones_like(s, dtype=bool)

    s_ext = s[extreme]
    u_ext = u[extreme]
    denom = float(np.dot(s_ext, s_ext))
    gamma_ratio = float(np.dot(s_ext, u_ext) / max(denom, 1e-12))
    gamma_ratio = float(np.clip(gamma_ratio, 0.05, 10.0))

    vel = u - gamma_ratio * s
    u_pred = gamma_ratio * s
    ss_res = float(np.sum((u[extreme] - u_pred[extreme]) ** 2))
    ss_tot = float(np.sum((u[extreme] - np.mean(u[extreme])) ** 2))
    r2 = float(np.clip(1.0 - ss_res / max(ss_tot, 1e-12), 0.0, 1.0))

    latent_g = np.clip(s / max(s_max, 1e-12), 0.0, 1.0)
    sigma = float(np.std(vel) + 1e-6)
    ll = -0.5 * len(s) * (np.log(2.0 * np.pi * sigma ** 2) + 1.0)

    fit = GeneVelocityFit(
        gene_name=gene_name,
        gene_index=gene_index,
        alpha=float(np.percentile(u, 95) * 1.0),
        beta=1.0,
        gamma=gamma_ratio,
        t_switch=1.0,
        r_squared=r2,
        log_likelihood=float(ll),
        noise_sigma=sigma,
        mode="steady_state",
    )
    return vel, s.copy(), u.copy(), latent_g, fit


def _assign_latent_time_grid(
    u_obs: FloatArray,
    s_obs: FloatArray,
    alpha: float,
    beta: float,
    gamma: float,
    t_switch: float,
    t_max: float,
    n_grid: int = 160,
) -> Tuple[FloatArray, FloatArray, FloatArray, float]:
    """E-step: assign optimal continuous latent time per cell along analytical curve."""
    t_grid = np.linspace(0.0, t_max, n_grid)
    u_grid, s_grid, _ = analytical_splicing_trajectory(t_grid, alpha, beta, gamma, t_switch)

    u_scale = max(float(np.std(u_obs)), 1e-3)
    s_scale = max(float(np.std(s_obs)), 1e-3)

    du = (u_obs[:, None] - u_grid[None, :]) / u_scale
    ds = (s_obs[:, None] - s_grid[None, :]) / s_scale
    err_sq = du * du + ds * ds
    best_idx = np.argmin(err_sq, axis=1)

    t_hat = t_grid[best_idx]
    u_hat, s_hat, _ = analytical_splicing_trajectory(t_hat, alpha, beta, gamma, t_switch)
    rss = float(np.mean(err_sq[np.arange(len(u_obs)), best_idx]))
    return t_hat, u_hat, s_hat, rss


def _fit_dynamical_gene(
    u: FloatArray,
    s: FloatArray,
    gene_name: str,
    gene_index: int,
    n_em_iters: int = 4,
) -> Tuple[FloatArray, FloatArray, FloatArray, FloatArray, GeneVelocityFit]:
    """Fit full dynamical splicing ODE parameters via EM + Nelder-Mead."""
    s_max = float(np.max(s))
    u_max = float(np.max(u))
    if s_max < 1e-6 or u_max < 1e-6 or float(np.std(s)) < 1e-6:
        return _fit_steady_state_gene(u, s, gene_name, gene_index)

    # Initialize from steady-state ratio
    _, _, _, _, ss_fit = _fit_steady_state_gene(u, s, gene_name, gene_index)
    beta_0 = 1.0
    gamma_0 = float(np.clip(ss_fit.gamma, 0.15, 4.0))
    alpha_0 = float(max(u_max * beta_0 * 1.15, 0.1))
    t_sw_0 = float(np.clip(2.2 / beta_0, 0.5, 5.0))
    t_max_0 = t_sw_0 + float(np.clip(2.5 / gamma_0, 0.8, 6.0))

    params = np.array([np.log(alpha_0), np.log(beta_0), np.log(gamma_0), np.log(t_sw_0)])
    t_hat, u_hat, s_hat, _ = _assign_latent_time_grid(
        u, s, alpha_0, beta_0, gamma_0, t_sw_0, t_max_0
    )

    u_scale = max(float(np.std(u)), 1e-3)
    s_scale = max(float(np.std(s)), 1e-3)

    for _ in range(n_em_iters):
        # M-step: optimize kinetic parameters holding t_hat fixed
        def objective(log_theta: FloatArray) -> float:
            a = float(np.exp(np.clip(log_theta[0], -8.0, 6.0)))
            b = float(np.exp(np.clip(log_theta[1], -5.0, 4.0)))
            g = float(np.exp(np.clip(log_theta[2], -5.0, 4.0)))
            tsw = float(np.exp(np.clip(log_theta[3], -3.0, 4.0)))
            u_m, s_m, _ = analytical_splicing_trajectory(t_hat, a, b, g, tsw)
            err = ((u - u_m) / u_scale) ** 2 + ((s - s_m) / s_scale) ** 2
            return float(np.mean(err))

        res = minimize(
            objective,
            params,
            method="Nelder-Mead",
            options={"maxiter": 65, "xatol": 1e-2, "fatol": 1e-3},
        )
        params = np.asarray(res.x, dtype=np.float64)
        alpha_cur = float(np.exp(np.clip(params[0], -8.0, 6.0)))
        beta_cur = float(np.exp(np.clip(params[1], -5.0, 4.0)))
        gamma_cur = float(np.exp(np.clip(params[2], -5.0, 4.0)))
        t_sw_cur = float(np.exp(np.clip(params[3], -3.0, 4.0)))
        t_max_cur = t_sw_cur + float(np.clip(3.0 / gamma_cur, 1.0, 8.0))

        # E-step: re-assign cell latent time along updated curve
        t_hat, u_hat, s_hat, _ = _assign_latent_time_grid(
            u, s, alpha_cur, beta_cur, gamma_cur, t_sw_cur, t_max_cur
        )

    alpha_f = float(np.exp(np.clip(params[0], -8.0, 6.0)))
    beta_f = float(np.exp(np.clip(params[1], -5.0, 4.0)))
    gamma_f = float(np.exp(np.clip(params[2], -5.0, 4.0)))
    t_sw_f = float(np.exp(np.clip(params[3], -3.0, 4.0)))

    # Instantaneous spliced velocity ds/dt = beta * u_hat - gamma * s_hat
    vel = beta_f * u - gamma_f * s

    ss_res = float(np.sum(((u - u_hat) / u_scale) ** 2 + ((s - s_hat) / s_scale) ** 2))
    ss_tot = float(
        np.sum(((u - np.mean(u)) / u_scale) ** 2 + ((s - np.mean(s)) / s_scale) ** 2)
    )
    r2 = float(np.clip(1.0 - ss_res / max(ss_tot, 1e-12), 0.0, 1.0))

    sigma_sq = max(ss_res / max(2 * len(u), 1), 1e-8)
    ll = -float(len(u)) * (np.log(2.0 * np.pi * sigma_sq) + 1.0)
    latent_norm = t_hat / max(float(np.max(t_hat)), 1e-12)

    fit = GeneVelocityFit(
        gene_name=gene_name,
        gene_index=gene_index,
        alpha=alpha_f,
        beta=beta_f,
        gamma=gamma_f,
        t_switch=t_sw_f,
        r_squared=r2,
        log_likelihood=ll,
        noise_sigma=float(np.sqrt(sigma_sq)),
        mode="dynamical",
    )
    return vel, s_hat, u_hat, latent_norm, fit


def compute_velocity_transition_matrix(
    spliced: FloatArray,
    velocities: FloatArray,
    connectivities: FloatArray,
    sigma_corr: float = 0.05,
) -> FloatArray:
    """Compute Boltzmann cosine-correlation velocity transition matrix ``pi_{ij}``.

    For each cell ``i`` and neighbor ``j`` with ``connectivities[i, j] > 0``,
    evaluates ``cos_ij = cos(v_i, s_j - s_i)`` and sets
    ``pi_{ij} = exp(cos_ij / sigma_corr) / sum_k exp(cos_ik / sigma_corr)``.
    Every row strictly sums to 1.0.
    """
    n_cells = spliced.shape[0]
    transition = np.zeros((n_cells, n_cells), dtype=np.float64)
    scale = 1.0 / max(sigma_corr, 1e-4)

    for i in range(n_cells):
        nbrs = np.flatnonzero(connectivities[i] > 1e-12)
        if len(nbrs) == 0:
            transition[i, i] = 1.0
            continue
        v_i = velocities[i]
        v_norm = float(np.linalg.norm(v_i))
        if v_norm < 1e-12:
            transition[i, nbrs] = 1.0 / float(len(nbrs))
            continue

        disp = spliced[nbrs] - spliced[i][None, :]
        disp_norms = np.linalg.norm(disp, axis=1)
        valid = disp_norms > 1e-12
        cosines = np.zeros(len(nbrs), dtype=np.float64)
        cosines[valid] = (disp[valid] @ v_i) / (disp_norms[valid] * v_norm)
        cosines = np.clip(cosines, -1.0, 1.0)

        logits = cosines * scale
        logits -= np.max(logits)
        weights = connectivities[i, nbrs] * np.exp(logits)
        w_sum = float(np.sum(weights))
        if w_sum <= 1e-18:
            transition[i, nbrs] = 1.0 / float(len(nbrs))
        else:
            transition[i, nbrs] = weights / w_sum

    return transition


def project_velocity_to_embedding(
    embedding: FloatArray,
    transition_matrix: FloatArray,
    connectivities: FloatArray,
) -> FloatArray:
    """Project high-dimensional cell velocity transitions onto 2D manifold embedding."""
    n_cells, n_dims = embedding.shape
    emb_vel = np.zeros((n_cells, n_dims), dtype=np.float64)
    for i in range(n_cells):
        nbrs = np.flatnonzero(connectivities[i] > 1e-12)
        if len(nbrs) == 0:
            continue
        disp = embedding[nbrs] - embedding[i][None, :]
        disp_norms = np.maximum(np.linalg.norm(disp, axis=1, keepdims=True), 1e-12)
        unit_disp = disp / disp_norms
        p_diff = transition_matrix[i, nbrs] - (1.0 / float(len(nbrs)))
        emb_vel[i] = np.sum(p_diff[:, None] * unit_disp, axis=0)
    return emb_vel


def compute_velocity_confidence(
    velocities: FloatArray,
    connectivities: FloatArray,
) -> FloatArray:
    """Measure local cosine coherence of velocity vectors across neighboring cells."""
    n_cells = velocities.shape[0]
    norms = np.linalg.norm(velocities, axis=1)
    confidence = np.zeros(n_cells, dtype=np.float64)

    for i in range(n_cells):
        if norms[i] < 1e-12:
            continue
        nbrs = np.flatnonzero(connectivities[i] > 1e-12)
        if len(nbrs) == 0:
            continue
        valid_nbrs = nbrs[norms[nbrs] > 1e-12]
        if len(valid_nbrs) == 0:
            continue
        cos_vals = (velocities[valid_nbrs] @ velocities[i]) / (norms[valid_nbrs] * norms[i])
        confidence[i] = float(np.clip(np.mean(cos_vals), -1.0, 1.0))
    return confidence


def estimate_rna_velocity(
    dataset: TranscriptomeDataset,
    mode: str = "dynamical",
    use_imputed: bool = True,
    min_r2_for_graph: float = 0.05,
    sigma_corr: float = 0.08,
) -> VelocityResult:
    """Estimate genome-wide RNA velocity, latent time, and 2D flow projection.

    Parameters
    ----------
    dataset:
        ``TranscriptomeDataset`` with ``spliced`` and ``unspliced`` (or
        MAGIC-imputed ``Ms`` and ``Mu``) layers.
    mode:
        ``"steady_state"`` or ``"dynamical"``.
    use_imputed:
        Prefer smoothed ``Ms``/``Mu`` layers when present.
    min_r2_for_graph:
        Minimum gene fit ``R^2`` required for inclusion in velocity transition graph.
    sigma_corr:
        Boltzmann temperature scaling cosine transition probabilities.
    """
    if mode not in ("steady_state", "dynamical"):
        raise ValueError(f"Unsupported velocity mode '{mode}'; expected 'steady_state' or 'dynamical'")

    if use_imputed and "Ms" in dataset.layers and "Mu" in dataset.layers:
        s_mat = dataset.layers["Ms"]
        u_mat = dataset.layers["Mu"]
    else:
        s_mat = dataset.layers["spliced"]
        u_mat = dataset.layers["unspliced"]

    n_cells, n_genes = s_mat.shape
    velocities = np.zeros((n_cells, n_genes), dtype=np.float64)
    s_hat_mat = np.zeros((n_cells, n_genes), dtype=np.float64)
    u_hat_mat = np.zeros((n_cells, n_genes), dtype=np.float64)
    latent_genes = np.zeros((n_cells, n_genes), dtype=np.float64)
    gene_fits: List[GeneVelocityFit] = []

    for g_idx, g_name in enumerate(dataset.gene_names):
        u_g = u_mat[:, g_idx]
        s_g = s_mat[:, g_idx]
        if mode == "steady_state":
            vel_g, sh_g, uh_g, lat_g, fit = _fit_steady_state_gene(u_g, s_g, g_name, g_idx)
        else:
            vel_g, sh_g, uh_g, lat_g, fit = _fit_dynamical_gene(u_g, s_g, g_name, g_idx)

        velocities[:, g_idx] = vel_g
        s_hat_mat[:, g_idx] = sh_g
        u_hat_mat[:, g_idx] = uh_g
        latent_genes[:, g_idx] = lat_g
        gene_fits.append(fit)

    r2_arr = np.array([f.r_squared for f in gene_fits], dtype=np.float64)
    good_genes = r2_arr >= min_r2_for_graph
    if int(np.sum(good_genes)) < 3:
        good_genes = np.ones(n_genes, dtype=bool)

    # Gene-shared latent time weighted by gene fit R^2
    weights = np.where(good_genes, np.maximum(r2_arr, 1e-3), 0.0)
    if float(np.sum(weights)) > 1e-12:
        latent_time = np.sum(latent_genes * weights[None, :], axis=1) / float(np.sum(weights))
    else:
        latent_time = np.mean(latent_genes, axis=1)
    # Rescale latent time cleanly to [0, 1]
    lt_min = float(np.min(latent_time))
    lt_max = float(np.max(latent_time))
    if lt_max - lt_min > 1e-12:
        latent_time = (latent_time - lt_min) / (lt_max - lt_min)

    if "connectivities" in dataset.obsp:
        conn = dataset.obsp["connectivities"]
    else:
        conn = np.ones((n_cells, n_cells), dtype=np.float64)
        np.fill_diagonal(conn, 0.0)

    transition = compute_velocity_transition_matrix(
        s_mat[:, good_genes],
        velocities[:, good_genes],
        conn,
        sigma_corr=sigma_corr,
    )
    confidence = compute_velocity_confidence(velocities[:, good_genes], conn)

    if "X_umap" in dataset.obsm:
        emb = dataset.obsm["X_umap"]
    elif "spatial" in dataset.obsm:
        emb = dataset.obsm["spatial"][:, :2]
    else:
        emb = np.zeros((n_cells, 2), dtype=np.float64)

    emb_vel = project_velocity_to_embedding(emb, transition, conn)

    dataset.set_layer("velocity", velocities, allow_negative=True)
    dataset.set_obsp("velocity_transition", transition)
    dataset.obs["latent_time"] = latent_time
    dataset.obs["velocity_confidence"] = confidence
    dataset.set_obsm("velocity_umap", emb_vel)

    return VelocityResult(
        velocities=velocities,
        spliced_hat=s_hat_mat,
        unspliced_hat=u_hat_mat,
        latent_time_genes=latent_genes,
        latent_time=latent_time,
        velocity_confidence=confidence,
        transition_matrix=transition,
        embedding_velocity=emb_vel,
        gene_fits=gene_fits,
        mode=mode,
        mean_r_squared=float(np.mean(r2_arr)),
        mean_confidence=float(np.mean(confidence)),
    )
