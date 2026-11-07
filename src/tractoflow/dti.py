"""
Diffusion Tensor Imaging (DTI) core tensor estimation and invariants calculation.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray


@dataclass
class DTIFitResult:
    """Container for fitted Diffusion Tensor volume results."""
    tensors: NDArray[np.float64]        # (X, Y, Z, 3, 3) symmetric positive semi-definite diffusion tensors
    eigenvalues: NDArray[np.float64]    # (X, Y, Z, 3) sorted descending lambda_1 >= lambda_2 >= lambda_3
    eigenvectors: NDArray[np.float64]   # (X, Y, Z, 3, 3) eigenvectors [:, :, :, :, i] matches lambda_i
    fa: NDArray[np.float64]             # (X, Y, Z) Fractional Anisotropy [0, 1]
    md: NDArray[np.float64]             # (X, Y, Z) Mean Diffusivity (Trace / 3)
    rd: NDArray[np.float64]             # (X, Y, Z) Radial Diffusivity ((lambda_2 + lambda_3) / 2)
    ad: NDArray[np.float64]             # (X, Y, Z) Axial Diffusivity (lambda_1)
    s0: NDArray[np.float64]             # (X, Y, Z) baseline b=0 signal


def construct_design_matrix(bvecs: NDArray[np.float64], bvals: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Construct Stejskal-Tanner linear design matrix B for DTI ordinary least squares (OLS).

    For diffusion gradient g = [gx, gy, gz]^T with b-value b:
    ln(S_k / S_0) = -b * (gx^2 * Dxx + 2*gx*gy * Dxy + 2*gx*gz * Dxz + gy^2 * Dyy + 2*gy*gz * Dyz + gz^2 * Dzz)

    Design matrix row: [-b*gx^2, -2*b*gx*gy, -2*b*gx*gz, -b*gy^2, -2*b*gy*gz, -b*gz^2, 1.0]
    Estimating: [Dxx, Dxy, Dxz, Dyy, Dyz, Dzz, ln(S0)]^T
    """
    K = len(bvals)
    B = np.zeros((K, 7), dtype=np.float64)
    for k in range(K):
        b = bvals[k]
        gx, gy, gz = bvecs[k]
        B[k, 0] = -b * (gx ** 2)
        B[k, 1] = -2.0 * b * (gx * gy)
        B[k, 2] = -2.0 * b * (gx * gz)
        B[k, 3] = -b * (gy ** 2)
        B[k, 4] = -2.0 * b * (gy * gz)
        B[k, 5] = -b * (gz ** 2)
        B[k, 6] = 1.0
    return B


def fit_dti_volume(
    dwi: NDArray[np.float64],
    bvecs: NDArray[np.float64],
    bvals: NDArray[np.float64],
    mask: NDArray[np.bool_] | None = None,
    clamp_negative_evals: bool = True
) -> DTIFitResult:
    """
    Fit Diffusion Tensor Imaging (DTI) model to 4D DWI dataset using Multivariate Linear Least Squares.

    Args:
        dwi: 4D volume of shape (Nx, Ny, Nz, K) with diffusion weighted signal
        bvecs: (K, 3) normalized gradient unit vectors
        bvals: (K,) b-values (e.g. 0 for baseline, 1000 for DWIs)
        mask: Optional binary brain mask (Nx, Ny, Nz)
        clamp_negative_evals: If True, enforce positive semi-definiteness lambda >= 1e-7

    Returns:
        DTIFitResult with tensors, eigenvalues, eigenvectors, FA, MD, RD, AD, S0.
    """
    shape = dwi.shape[:3]
    Nx, Ny, Nz = shape
    K = dwi.shape[3]

    if mask is None:
        # auto mask voxels with non-zero baseline signal
        b0_idx = np.where(bvals < 50)[0]
        if len(b0_idx) > 0:
            mean_b0 = np.mean(dwi[..., b0_idx], axis=-1)
            mask = mean_b0 > (0.05 * np.max(mean_b0))
        else:
            mask = np.ones(shape, dtype=bool)

    # Construct design matrix and pseudo-inverse
    B = construct_design_matrix(bvecs, bvals)  # (K, 7)
    B_pinv = np.linalg.pinv(B)                # (7, K)

    # Pre-allocate output arrays
    tensors = np.zeros((*shape, 3, 3), dtype=np.float64)
    eigenvalues = np.zeros((*shape, 3), dtype=np.float64)
    eigenvectors = np.zeros((*shape, 3, 3), dtype=np.float64)
    fa = np.zeros(shape, dtype=np.float64)
    md = np.zeros(shape, dtype=np.float64)
    rd = np.zeros(shape, dtype=np.float64)
    ad = np.zeros(shape, dtype=np.float64)
    s0 = np.zeros(shape, dtype=np.float64)

    # Masked coordinates
    voxel_indices = np.argwhere(mask)
    if len(voxel_indices) == 0:
        return DTIFitResult(tensors, eigenvalues, eigenvectors, fa, md, rd, ad, s0)

    # Extract signals: (N_voxels, K)
    signals = dwi[mask]  # shape (M, K)
    # Avoid log(0) / negative noise
    eps = 1e-4
    signals = np.maximum(signals, eps)
    log_signals = np.log(signals)  # (M, K)

    # Solve beta = B_pinv @ log_S^T -> shape (7, M)
    beta = B_pinv @ log_signals.T  # (7, M)
    beta = beta.T                  # (M, 7)

    # Unpack parameters: [Dxx, Dxy, Dxz, Dyy, Dyz, Dzz, ln_S0]
    dxx = beta[:, 0]
    dxy = beta[:, 1]
    dxz = beta[:, 2]
    dyy = beta[:, 3]
    dyz = beta[:, 4]
    dzz = beta[:, 5]
    s0_vals = np.exp(beta[:, 6])

    # Construct 3x3 symmetric matrix for each voxel
    M = len(voxel_indices)
    tensor_vox = np.zeros((M, 3, 3), dtype=np.float64)
    tensor_vox[:, 0, 0] = dxx
    tensor_vox[:, 0, 1] = dxy
    tensor_vox[:, 1, 0] = dxy
    tensor_vox[:, 0, 2] = dxz
    tensor_vox[:, 2, 0] = dxz
    tensor_vox[:, 1, 1] = dyy
    tensor_vox[:, 1, 2] = dyz
    tensor_vox[:, 2, 1] = dyz
    tensor_vox[:, 2, 2] = dzz

    # Eigendecomposition for symmetric matrices (eigh)
    evals, evecs = np.linalg.eigh(tensor_vox)  # evals ascending (M, 3), evecs (M, 3, 3)

    # Sort descending: lambda_1 >= lambda_2 >= lambda_3
    evals = evals[:, ::-1]
    evecs = evecs[:, :, ::-1]

    if clamp_negative_evals:
        evals = np.maximum(evals, 1e-7)
        # Reconstruct clamped tensor: V * Lambda * V^T
        # einsum for batch matrix multiplication: (M, 3, 3) @ diag(evals) @ (M, 3, 3)^T
        tensor_vox = np.einsum('mij,mj,mkj->mik', evecs, evals, evecs)

    l1 = evals[:, 0]
    l2 = evals[:, 1]
    l3 = evals[:, 2]

    md_vals = (l1 + l2 + l3) / 3.0
    ad_vals = l1
    rd_vals = (l2 + l3) / 2.0

    # Fractional Anisotropy (FA)
    # FA = sqrt(3/2) * sqrt((l1-MD)^2 + (l2-MD)^2 + (l3-MD)^2) / sqrt(l1^2 + l2^2 + l3^2)
    var_l = (l1 - md_vals)**2 + (l2 - md_vals)**2 + (l3 - md_vals)**2
    sum_sq_l = l1**2 + l2**2 + l3**2
    with np.errstate(divide='ignore', invalid='ignore'):
        fa_vals = np.sqrt(1.5) * np.sqrt(var_l) / np.sqrt(np.maximum(sum_sq_l, 1e-12))
        fa_vals = np.nan_to_num(fa_vals, nan=0.0, posinf=0.0, neginf=0.0)
        fa_vals = np.clip(fa_vals, 0.0, 1.0)

    # Assign back into 3D grids
    for idx, (x, y, z) in enumerate(voxel_indices):
        tensors[x, y, z] = tensor_vox[idx]
        eigenvalues[x, y, z] = evals[idx]
        eigenvectors[x, y, z] = evecs[idx]
        fa[x, y, z] = fa_vals[idx]
        md[x, y, z] = md_vals[idx]
        rd[x, y, z] = rd_vals[idx]
        ad[x, y, z] = ad_vals[idx]
        s0[x, y, z] = s0_vals[idx]

    return DTIFitResult(
        tensors=tensors,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        fa=fa,
        md=md,
        rd=rd,
        ad=ad,
        s0=s0
    )
