"""
Connectome Graph Extraction and Structural Connectivity Analysis.
Maps fiber streamlines intersecting anatomical parcellation regions into
weighted adjacency matrices and computes network topology invariants.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from .tractography import Tractogram, Streamline


@dataclass
class ConnectomeGraph:
    adjacency: NDArray[np.float64]         # (N_nodes, N_nodes) edge weights (fiber count or density)
    mean_fa_matrix: NDArray[np.float64]   # (N_nodes, N_nodes) average FA along connecting fibers
    node_degrees: NDArray[np.float64]      # (N_nodes,)
    node_strengths: NDArray[np.float64]    # (N_nodes,) sum of incident edge weights
    global_efficiency: float               # network integration metric
    density: float                         # ratio of active connections to possible connections
    clustering_coefficient: float          # average clustering coefficient


def build_parcellation_atlas(
    grid_size: tuple[int, int, int] = (32, 32, 20),
    n_regions: int = 6
) -> NDArray[np.int32]:
    """
    Create an anatomical volume parcellation map where each voxel is assigned
    a Region-Of-Interest (ROI) ID: 1..n_regions (0 is non-ROI tissue/background).
    """
    Nx, Ny, Nz = grid_size
    atlas = np.zeros(grid_size, dtype=np.int32)

    # Divide volume into symmetric cortical / subcortical bounding ROIs
    # Region 1: Superior Left (Z > Nz/2, X < Nx/2)
    # Region 2: Superior Right (Z > Nz/2, X >= Nx/2)
    # Region 3: Inferior Left (Z <= Nz/2, X < Nx/2)
    # Region 4: Inferior Right (Z <= Nz/2, X >= Nx/2)
    # Region 5: Anterior Center (Y > 3*Ny/4)
    # Region 6: Posterior Center (Y < Ny/4)

    mid_x, mid_y, mid_z = Nx // 2, Ny // 2, Nz // 2

    atlas[:mid_x, :, mid_z:] = 1
    atlas[mid_x:, :, mid_z:] = 2
    atlas[:mid_x, :, :mid_z] = 3
    atlas[mid_x:, :, :mid_z] = 4

    # Overwrite poles
    if n_regions >= 5:
        atlas[:, int(3*Ny/4):, :] = 5
    if n_regions >= 6:
        atlas[:, :int(Ny/4), :] = 6

    return atlas


def extract_connectome(
    tractogram: Tractogram,
    parcellation: NDArray[np.int32],
    n_regions: int = 6
) -> ConnectomeGraph:
    """
    Identify end-point ROI intersections for each streamline and compute
    structural connectome matrices and graph-theoretic metrics.
    """
    adjacency = np.zeros((n_regions, n_regions), dtype=np.float64)
    fa_accum = np.zeros((n_regions, n_regions), dtype=np.float64)
    count_matrix = np.zeros((n_regions, n_regions), dtype=np.float64)

    Nx, Ny, Nz = parcellation.shape

    def get_roi_at_point(pt: NDArray[np.float64]) -> int:
        x, y, z = int(round(pt[0])), int(round(pt[1])), int(round(pt[2]))
        if 0 <= x < Nx and 0 <= y < Ny and 0 <= z < Nz:
            return int(parcellation[x, y, z])
        return 0

    for sl in tractogram.streamlines:
        if len(sl.points) < 2:
            continue

        roi_start = get_roi_at_point(sl.points[0])
        roi_end = get_roi_at_point(sl.points[-1])

        # Only count connections between valid ROIs (1-indexed)
        if 1 <= roi_start <= n_regions and 1 <= roi_end <= n_regions:
            i = roi_start - 1
            j = roi_end - 1
            # Undirected graph
            mean_sl_fa = float(np.mean(sl.fa_profile)) if len(sl.fa_profile) > 0 else 0.0
            adjacency[i, j] += 1.0
            fa_accum[i, j] += mean_sl_fa
            count_matrix[i, j] += 1.0
            if i != j:
                adjacency[j, i] += 1.0
                fa_accum[j, i] += mean_sl_fa
                count_matrix[j, i] += 1.0

    # Mean FA along connections
    with np.errstate(divide='ignore', invalid='ignore'):
        mean_fa_matrix = np.where(count_matrix > 0, fa_accum / count_matrix, 0.0)

    # Graph Metrics
    node_strengths = np.sum(adjacency, axis=1)
    node_degrees = np.sum(adjacency > 0, axis=1).astype(np.float64)

    total_possible_edges = n_regions * (n_regions - 1)
    active_edges = np.sum(adjacency > 0) - np.trace(adjacency > 0)
    density = float(active_edges / max(1, total_possible_edges))

    # Clustering coefficient
    # C_i = 2 * e_i / (k_i * (k_i - 1))
    binary_adj = (adjacency > 0).astype(float)
    np.fill_diagonal(binary_adj, 0)
    cc_list = []
    for i in range(n_regions):
        neighbors = np.where(binary_adj[i] > 0)[0]
        k = len(neighbors)
        if k < 2:
            cc_list.append(0.0)
        else:
            subgraph = binary_adj[np.ix_(neighbors, neighbors)]
            e = np.sum(subgraph) / 2.0
            cc_list.append(2.0 * e / (k * (k - 1)))
    clustering_coefficient = float(np.mean(cc_list)) if cc_list else 0.0

    # Global efficiency via shortest path matrix (Floyd-Warshall)
    # Distance: d_ij = 1 / W_ij (or infinity if no connection)
    dist_matrix = np.full((n_regions, n_regions), np.inf)
    for i in range(n_regions):
        dist_matrix[i, i] = 0.0
        for j in range(n_regions):
            if i != j and adjacency[i, j] > 0:
                dist_matrix[i, j] = 1.0 / adjacency[i, j]

    # Floyd-Warshall
    for k in range(n_regions):
        for i in range(n_regions):
            for j in range(n_regions):
                if dist_matrix[i, k] + dist_matrix[k, j] < dist_matrix[i, j]:
                    dist_matrix[i, j] = dist_matrix[i, k] + dist_matrix[k, j]

    inv_distances = []
    for i in range(n_regions):
        for j in range(n_regions):
            if i != j and not np.isinf(dist_matrix[i, j]):
                inv_distances.append(1.0 / dist_matrix[i, j])
            elif i != j:
                inv_distances.append(0.0)

    global_efficiency = float(np.mean(inv_distances)) if inv_distances else 0.0

    return ConnectomeGraph(
        adjacency=adjacency,
        mean_fa_matrix=mean_fa_matrix,
        node_degrees=node_degrees,
        node_strengths=node_strengths,
        global_efficiency=global_efficiency,
        density=density,
        clustering_coefficient=clustering_coefficient
    )
