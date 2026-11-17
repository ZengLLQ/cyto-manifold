"""Self-contained AnnData-inspired sparse/dense transcriptome container.

Provides ``TranscriptomeDataset`` for storing spliced and unspliced UMI count
layers, normalized and imputed expression matrices, 2D spatial coordinates,
low-dimensional embeddings (``obsm``), cell-cell adjacency graphs (``obsp``),
per-cell metadata (``obs``), and per-gene metadata (``var``), with lossless
NPZ/JSON serialization and SHA-256 integrity verification.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
from numpy.typing import NDArray
from scipy import sparse

from cyto_manifold.types import (
    BoolArray,
    FloatArray,
    validate_matrix_shape,
    validate_non_negative,
)

MatrixLike = Union[NDArray[np.float64], sparse.spmatrix]


def _to_dense_float64(mat: MatrixLike) -> FloatArray:
    """Convert dense or scipy sparse matrix to contiguous float64 ndarray."""
    if sparse.issparse(mat):
        sp_mat: Any = mat
        return np.asarray(sp_mat.toarray(), dtype=np.float64)
    return np.asarray(mat, dtype=np.float64)


class TranscriptomeDataset:
    """Multi-layer single-cell and spatial transcriptomics data container.

    Attributes
    ----------
    X:
        Primary expression matrix of shape ``(n_cells, n_genes)``.
    cell_ids:
        Unique barcode identifiers of length ``n_cells``.
    gene_names:
        Gene symbols of length ``n_genes``.
    layers:
        Named expression layers such as ``spliced``, ``unspliced``,
        ``raw_counts``, ``normalized``, and ``imputed``.
    obs:
        Column-oriented per-cell metadata dictionary.
    var:
        Column-oriented per-gene metadata dictionary.
    obsm:
        Multi-dimensional per-cell embeddings (``X_pca``, ``X_umap``,
        ``X_diffmap``, ``spatial``).
    obsp:
        Pairwise ``(n_cells, n_cells)`` graphs (``connectivities``,
        ``distances``, ``velocity_transition``, ``spatial_weights``).
    uns:
        Unstructured metadata dictionary.
    """

    def __init__(
        self,
        X: MatrixLike,
        cell_ids: Optional[Sequence[str]] = None,
        gene_names: Optional[Sequence[str]] = None,
        spliced: Optional[MatrixLike] = None,
        unspliced: Optional[MatrixLike] = None,
        spatial: Optional[NDArray[np.float64]] = None,
        obs: Optional[Mapping[str, Union[Sequence[Any], NDArray[Any]]]] = None,
        var: Optional[Mapping[str, Union[Sequence[Any], NDArray[Any]]]] = None,
        uns: Optional[Dict[str, Any]] = None,
    ) -> None:
        dense_x = _to_dense_float64(X)
        if dense_x.ndim != 2:
            raise ValueError(f"Primary matrix X must be 2D, got ndim={dense_x.ndim}")
        n_cells, n_genes = dense_x.shape
        if n_cells == 0 or n_genes == 0:
            raise ValueError("TranscriptomeDataset cannot have zero cells or zero genes")
        if not np.all(np.isfinite(dense_x)):
            raise ValueError("Primary matrix X contains non-finite values")
        validate_non_negative(dense_x, "X")

        self.X: FloatArray = dense_x.copy()
        self.cell_ids: List[str] = (
            list(cell_ids)
            if cell_ids is not None
            else [f"CELL_{i:05d}" for i in range(n_cells)]
        )
        self.gene_names: List[str] = (
            list(gene_names)
            if gene_names is not None
            else [f"GENE_{j:04d}" for j in range(n_genes)]
        )

        if len(self.cell_ids) != n_cells:
            raise ValueError("Length of cell_ids must match n_cells")
        if len(self.gene_names) != n_genes:
            raise ValueError("Length of gene_names must match n_genes")
        if len(set(self.cell_ids)) != n_cells:
            raise ValueError("cell_ids must be unique")
        if len(set(self.gene_names)) != n_genes:
            raise ValueError("gene_names must be unique")

        self.layers: Dict[str, FloatArray] = {
            "raw_counts": self.X.copy(),
        }
        if spliced is not None:
            s_arr = validate_matrix_shape(_to_dense_float64(spliced), n_cells, n_genes, "spliced")
            validate_non_negative(s_arr, "spliced")
            self.layers["spliced"] = s_arr
        else:
            self.layers["spliced"] = self.X.copy()

        if unspliced is not None:
            u_arr = validate_matrix_shape(
                _to_dense_float64(unspliced), n_cells, n_genes, "unspliced"
            )
            validate_non_negative(u_arr, "unspliced")
            self.layers["unspliced"] = u_arr
        else:
            self.layers["unspliced"] = np.zeros_like(self.X)

        self.obs: Dict[str, NDArray[Any]] = {}
        if obs is not None:
            for key, col in obs.items():
                arr = np.asarray(list(col))
                if len(arr) != n_cells:
                    raise ValueError(f"obs['{key}'] length {len(arr)} != n_cells {n_cells}")
                self.obs[key] = arr

        self.var: Dict[str, NDArray[Any]] = {}
        if var is not None:
            for key, col in var.items():
                arr = np.asarray(list(col))
                if len(arr) != n_genes:
                    raise ValueError(f"var['{key}'] length {len(arr)} != n_genes {n_genes}")
                self.var[key] = arr

        self.obsm: Dict[str, FloatArray] = {}
        if spatial is not None:
            sp = np.asarray(spatial, dtype=np.float64)
            if sp.ndim != 2 or sp.shape[0] != n_cells or sp.shape[1] < 2:
                raise ValueError(
                    f"spatial coordinates must have shape ({n_cells}, >=2), got {sp.shape}"
                )
            if not np.all(np.isfinite(sp)):
                raise ValueError("spatial coordinates contain non-finite values")
            self.obsm["spatial"] = sp.copy()

        self.obsp: Dict[str, FloatArray] = {}
        self.uns: Dict[str, Any] = dict(uns) if uns is not None else {}

    @property
    def n_cells(self) -> int:
        """Return total number of cells/spots."""
        return int(self.X.shape[0])

    @property
    def n_genes(self) -> int:
        """Return total number of genes."""
        return int(self.X.shape[1])

    def has_spatial(self) -> bool:
        """Return True if 2D/3D spatial coordinates are attached."""
        return "spatial" in self.obsm

    def gene_index(self, gene_name: str) -> int:
        """Return integer column index for ``gene_name`` or raise KeyError."""
        try:
            return self.gene_names.index(gene_name)
        except ValueError as exc:
            raise KeyError(f"Gene '{gene_name}' not found in dataset") from exc

    def set_layer(self, name: str, values: MatrixLike, allow_negative: bool = True) -> None:
        """Store a validated layer matrix of shape ``(n_cells, n_genes)``."""
        arr = validate_matrix_shape(_to_dense_float64(values), self.n_cells, self.n_genes, name)
        if not allow_negative:
            validate_non_negative(arr, name)
        self.layers[name] = arr.copy()

    def set_obsm(self, name: str, embedding: NDArray[np.float64]) -> None:
        """Store a per-cell embedding array of shape ``(n_cells, n_dims)``."""
        arr = np.asarray(embedding, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] != self.n_cells:
            raise ValueError(
                f"obsm['{name}'] must have shape ({self.n_cells}, d), got {arr.shape}"
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"obsm['{name}'] contains non-finite values")
        self.obsm[name] = arr.copy()

    def set_obsp(self, name: str, graph: MatrixLike) -> None:
        """Store a pairwise cell graph of shape ``(n_cells, n_cells)``."""
        arr = validate_matrix_shape(
            _to_dense_float64(graph), self.n_cells, self.n_cells, f"obsp['{name}']"
        )
        self.obsp[name] = arr.copy()

    def to_sparse_layer(self, layer_name: str) -> sparse.csr_matrix:
        """Return a CSR sparse representation of a named layer."""
        if layer_name not in self.layers:
            raise KeyError(f"Unknown layer '{layer_name}'")
        return sparse.csr_matrix(self.layers[layer_name])

    def subset(
        self,
        cell_mask: Optional[BoolArray] = None,
        gene_mask: Optional[BoolArray] = None,
    ) -> "TranscriptomeDataset":
        """Create a deep-copied subset filtered by boolean cell and gene masks."""
        c_mask = (
            np.ones(self.n_cells, dtype=bool)
            if cell_mask is None
            else np.asarray(cell_mask, dtype=bool)
        )
        g_mask = (
            np.ones(self.n_genes, dtype=bool)
            if gene_mask is None
            else np.asarray(gene_mask, dtype=bool)
        )
        if c_mask.shape != (self.n_cells,):
            raise ValueError("cell_mask shape mismatch")
        if g_mask.shape != (self.n_genes,):
            raise ValueError("gene_mask shape mismatch")
        if not np.any(c_mask) or not np.any(g_mask):
            raise ValueError("Subset cannot produce zero cells or zero genes")

        c_idx = np.flatnonzero(c_mask)
        g_idx = np.flatnonzero(g_mask)

        new_ds = TranscriptomeDataset(
            X=self.layers["raw_counts"][np.ix_(c_idx, g_idx)],
            cell_ids=[self.cell_ids[i] for i in c_idx],
            gene_names=[self.gene_names[j] for j in g_idx],
            spliced=self.layers["spliced"][np.ix_(c_idx, g_idx)],
            unspliced=self.layers["unspliced"][np.ix_(c_idx, g_idx)],
            spatial=self.obsm["spatial"][c_idx] if "spatial" in self.obsm else None,
            obs={k: v[c_idx] for k, v in self.obs.items()},
            var={k: v[g_idx] for k, v in self.var.items()},
            uns=dict(self.uns),
        )
        new_ds.X = self.X[np.ix_(c_idx, g_idx)].copy()
        for lname, lmat in self.layers.items():
            if lname not in ("raw_counts", "spliced", "unspliced"):
                new_ds.layers[lname] = lmat[np.ix_(c_idx, g_idx)].copy()
        for mname, marr in self.obsm.items():
            if mname != "spatial":
                new_ds.obsm[mname] = marr[c_idx].copy()
        for pname, pmat in self.obsp.items():
            new_ds.obsp[pname] = pmat[np.ix_(c_idx, c_idx)].copy()
        return new_ds

    def copy(self) -> "TranscriptomeDataset":
        """Return an independent deep copy of the dataset."""
        return self.subset()

    def content_checksum(self) -> str:
        """Compute deterministic SHA-256 digest over matrix data and identifiers."""
        hasher = hashlib.sha256()
        hasher.update("|".join(self.cell_ids).encode("utf-8"))
        hasher.update(b"\n")
        hasher.update("|".join(self.gene_names).encode("utf-8"))
        hasher.update(np.ascontiguousarray(self.X, dtype=np.float64).tobytes())
        hasher.update(np.ascontiguousarray(self.layers["spliced"], dtype=np.float64).tobytes())
        hasher.update(np.ascontiguousarray(self.layers["unspliced"], dtype=np.float64).tobytes())
        if "spatial" in self.obsm:
            hasher.update(np.ascontiguousarray(self.obsm["spatial"], dtype=np.float64).tobytes())
        return hasher.hexdigest()

    def save_npz(self, path: Union[str, Path]) -> Path:
        """Persist dataset layers, spatial coordinates, and metadata to an ``.npz`` archive."""
        out_path = Path(path)
        payload: Dict[str, Any] = {
            "X": self.X,
            "cell_ids": np.asarray(self.cell_ids, dtype=str),
            "gene_names": np.asarray(self.gene_names, dtype=str),
            "layer_names": np.asarray(list(self.layers.keys()), dtype=str),
            "obsm_names": np.asarray(list(self.obsm.keys()), dtype=str),
            "obs_json": np.asarray(
                json.dumps({k: v.tolist() for k, v in self.obs.items()}), dtype=str
            ),
            "var_json": np.asarray(
                json.dumps({k: v.tolist() for k, v in self.var.items()}), dtype=str
            ),
            "uns_json": np.asarray(json.dumps(self.uns, default=str), dtype=str),
        }
        for lname, lmat in self.layers.items():
            payload[f"layer__{lname}"] = lmat
        for mname, marr in self.obsm.items():
            payload[f"obsm__{mname}"] = marr
        np.savez_compressed(out_path, **payload)
        return out_path

    @classmethod
    def load_npz(cls, path: Union[str, Path]) -> "TranscriptomeDataset":
        """Load a ``TranscriptomeDataset`` previously saved by ``save_npz``."""
        in_path = Path(path)
        with np.load(in_path, allow_pickle=False) as archive:
            x_mat = np.asarray(archive["X"], dtype=np.float64)
            cell_ids = [str(c) for c in archive["cell_ids"].tolist()]
            gene_names = [str(g) for g in archive["gene_names"].tolist()]
            obs_dict = json.loads(str(archive["obs_json"].item()))
            var_dict = json.loads(str(archive["var_json"].item()))
            uns_dict = json.loads(str(archive["uns_json"].item()))
            spliced = np.asarray(archive["layer__spliced"], dtype=np.float64)
            unspliced = np.asarray(archive["layer__unspliced"], dtype=np.float64)
            spatial = (
                np.asarray(archive["obsm__spatial"], dtype=np.float64)
                if "obsm__spatial" in archive
                else None
            )
            ds = cls(
                X=x_mat,
                cell_ids=cell_ids,
                gene_names=gene_names,
                spliced=spliced,
                unspliced=unspliced,
                spatial=spatial,
                obs=obs_dict,
                var=var_dict,
                uns=uns_dict,
            )
            for lname in archive["layer_names"].tolist():
                if lname not in ("raw_counts", "spliced", "unspliced"):
                    ds.layers[str(lname)] = np.asarray(archive[f"layer__{lname}"], dtype=np.float64)
            for mname in archive["obsm_names"].tolist():
                if mname != "spatial":
                    ds.obsm[str(mname)] = np.asarray(archive[f"obsm__{mname}"], dtype=np.float64)
            return ds
