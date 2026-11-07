"""
Comprehensive test suite for TractoFlow DTI Fiber Tractography Engine.
Covers: design matrix, DTI fitting invariance, FA/MD math, phantom physics,
RK4 integration, streamline stopping criteria, connectome graph metrics.
"""

from __future__ import annotations
import numpy as np
import pytest

from tractoflow.dti import (
    construct_design_matrix,
    fit_dti_volume,
    DTIFitResult,
)
from tractoflow.phantom import (
    create_curved_bundle_phantom,
    generate_gradient_scheme,
)
from tractoflow.tractography import (
    trilinear_interpolate_vector,
    trilinear_interpolate_scalar,
    rk4_step,
    track_streamline_bidirectional,
    track_ensemble,
    Streamline,
    Tractogram,
)
from tractoflow.connectome import (
    build_parcellation_atlas,
    extract_connectome,
    ConnectomeGraph,
)


# ─────────────────────────────────────────────────────────────────────────────
# Design Matrix & Gradient Scheme
# ─────────────────────────────────────────────────────────────────────────────

class TestDesignMatrix:
    def test_shape_and_b0_row(self):
        bvecs = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        bvals = np.array([0.0, 1000.0, 1000.0])
        B = construct_design_matrix(bvecs, bvals)
        assert B.shape == (3, 7)
        # b=0 row: only the intercept column is 1
        np.testing.assert_allclose(B[0, :6], 0.0)
        assert B[0, 6] == 1.0

    def test_x_direction_only_dxx(self):
        bvecs = np.array([[1.0, 0.0, 0.0]])
        bvals = np.array([1000.0])
        B = construct_design_matrix(bvecs, bvals)
        # Only Dxx coefficient nonzero among tensor terms: -b * gx^2 = -1000
        assert B[0, 0] == pytest.approx(-1000.0)
        np.testing.assert_allclose(B[0, 1:6], 0.0)

    def test_gradient_scheme_unit_norm(self):
        bvecs, bvals = generate_gradient_scheme(n_directions=20, b_value=1000.0, n_b0=3)
        assert len(bvecs) == 23
        assert np.sum(bvals == 0) == 3
        for v, b in zip(bvecs, bvals):
            if b > 0:
                assert np.linalg.norm(v) == pytest.approx(1.0, abs=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# DTI Fitting
# ─────────────────────────────────────────────────────────────────────────────

class TestDTIFit:
    def _make_single_fiber_volume(self):
        """Create a tiny 3x3x3 volume with pure Z-axis anisotropy."""
        d_par, d_perp = 1.7e-3, 0.3e-3
        v = np.array([0.0, 0.0, 1.0])
        D = d_par * np.outer(v, v) + d_perp * (np.eye(3) - np.outer(v, v))
        s0 = 1000.0

        bvecs, bvals = generate_gradient_scheme(n_directions=20, b_value=1000.0, n_b0=2)
        K = len(bvals)
        dwi = np.zeros((3, 3, 3, K))
        for k in range(K):
            g = bvecs[k]
            atten = np.exp(-bvals[k] * (g @ D @ g))
            dwi[:, :, :, k] = s0 * atten
        mask = np.ones((3, 3, 3), dtype=bool)
        return dwi, bvecs, bvals, mask, D

    def test_fit_recovers_principal_direction(self):
        dwi, bvecs, bvals, mask, D_true = self._make_single_fiber_volume()
        result = fit_dti_volume(dwi, bvecs, bvals, mask=mask)
        assert isinstance(result, DTIFitResult)
        # Principal eigenvector should align with Z (±)
        v1 = result.eigenvectors[1, 1, 1, :, 0]
        alignment = abs(np.dot(v1, [0, 0, 1]))
        assert alignment > 0.95, f"v1 alignment={alignment}"

    def test_fa_high_for_anisotropic_fiber(self):
        dwi, bvecs, bvals, mask, _ = self._make_single_fiber_volume()
        result = fit_dti_volume(dwi, bvecs, bvals, mask=mask)
        # Theoretical FA for lambda=[1.7, 0.3, 0.3]*1e-3
        fa_center = result.fa[1, 1, 1]
        assert fa_center > 0.6, f"FA too low: {fa_center}"

    def test_md_matches_trace(self):
        dwi, bvecs, bvals, mask, D_true = self._make_single_fiber_volume()
        result = fit_dti_volume(dwi, bvecs, bvals, mask=mask)
        md_true = np.trace(D_true) / 3.0
        md_est = result.md[1, 1, 1]
        assert md_est == pytest.approx(md_true, rel=0.05)

    def test_eigenvalues_sorted_descending(self):
        dwi, bvecs, bvals, mask, _ = self._make_single_fiber_volume()
        result = fit_dti_volume(dwi, bvecs, bvals, mask=mask)
        ev = result.eigenvalues[1, 1, 1]
        assert ev[0] >= ev[1] >= ev[2]

    def test_tensors_symmetric(self):
        dwi, bvecs, bvals, mask, _ = self._make_single_fiber_volume()
        result = fit_dti_volume(dwi, bvecs, bvals, mask=mask)
        T = result.tensors[1, 1, 1]
        np.testing.assert_allclose(T, T.T, atol=1e-12)

    def test_fa_bounded_0_1(self):
        dwi, bvecs, bvals, mask, _ = self._make_single_fiber_volume()
        result = fit_dti_volume(dwi, bvecs, bvals, mask=mask)
        assert np.all(result.fa >= 0.0)
        assert np.all(result.fa <= 1.0)

    def test_empty_mask_returns_zeros(self):
        dwi = np.ones((2, 2, 2, 4))
        bvecs = np.eye(3)
        bvecs = np.vstack([np.zeros(3), bvecs])
        bvals = np.array([0.0, 1000, 1000, 1000])
        mask = np.zeros((2, 2, 2), dtype=bool)
        result = fit_dti_volume(dwi, bvecs[:4], bvals, mask=mask)
        assert np.allclose(result.fa, 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Phantom
# ─────────────────────────────────────────────────────────────────────────────

class TestPhantom:
    def test_phantom_shapes(self):
        ph = create_curved_bundle_phantom(grid_size=(16, 16, 10), n_directions=12, seed=1)
        assert ph.dwi.shape[:3] == (16, 16, 10)
        assert ph.dwi.shape[3] == 12 + 4  # directions + b0s
        assert ph.mask.shape == (16, 16, 10)
        assert ph.dwi.min() >= 0  # Rician noise stays non-negative

    def test_phantom_has_high_fa_regions(self):
        ph = create_curved_bundle_phantom(grid_size=(20, 20, 12), snr=80.0, n_directions=24, seed=2)
        result = fit_dti_volume(ph.dwi, ph.bvecs, ph.bvals, mask=ph.mask)
        # Some voxels should have FA > 0.4 (anisotropic tracts)
        assert np.max(result.fa) > 0.4


# ─────────────────────────────────────────────────────────────────────────────
# Interpolation & RK4
# ─────────────────────────────────────────────────────────────────────────────

class TestInterpolation:
    def test_trilinear_vector_at_grid_point(self):
        field = np.zeros((4, 4, 4, 3))
        field[1, 1, 1] = [0.0, 0.0, 1.0]
        field[2, 1, 1] = [0.0, 0.0, 1.0]
        field[1, 2, 1] = [0.0, 0.0, 1.0]
        field[2, 2, 1] = [0.0, 0.0, 1.0]
        field[1, 1, 2] = [0.0, 0.0, 1.0]
        field[2, 1, 2] = [0.0, 0.0, 1.0]
        field[1, 2, 2] = [0.0, 0.0, 1.0]
        field[2, 2, 2] = [0.0, 0.0, 1.0]
        v = trilinear_interpolate_vector(field, np.array([1.5, 1.5, 1.5]))
        assert v is not None
        assert abs(np.dot(v, [0, 0, 1])) > 0.99

    def test_out_of_bounds_returns_none(self):
        field = np.ones((4, 4, 4, 3))
        assert trilinear_interpolate_vector(field, np.array([-1.0, 1.0, 1.0])) is None
        assert trilinear_interpolate_scalar(np.ones((4, 4, 4)), np.array([10.0, 1.0, 1.0])) is None

    def test_rk4_advances_along_field(self):
        # Uniform Z-directed field
        field = np.zeros((10, 10, 10, 3))
        field[..., 2] = 1.0
        pos = np.array([5.0, 5.0, 2.0])
        direction = np.array([0.0, 0.0, 1.0])
        new_pos, new_dir = rk4_step(field, pos, direction, step_size=1.0)
        assert new_pos is not None
        assert new_pos[2] > pos[2]  # advanced in +Z
        assert abs(np.dot(new_dir, [0, 0, 1])) > 0.9


# ─────────────────────────────────────────────────────────────────────────────
# Tractography
# ─────────────────────────────────────────────────────────────────────────────

class TestTractography:
    def _setup_straight_field(self):
        """Uniform Z-fiber field with high FA."""
        N = 12
        v1 = np.zeros((N, N, N, 3))
        v1[..., 2] = 1.0  # all point +Z
        fa = np.ones((N, N, N)) * 0.8
        return v1, fa

    def test_bidirectional_tracking_produces_streamline(self):
        v1, fa = self._setup_straight_field()
        seed = np.array([6.0, 6.0, 6.0])
        sl = track_streamline_bidirectional(
            seed=seed, v1_field=v1, fa_map=fa,
            step_size=0.5, fa_threshold=0.2, max_angle_deg=30.0,
            max_steps=100, method="rk4"
        )
        assert sl is not None
        assert isinstance(sl, Streamline)
        assert sl.length > 2.0
        assert len(sl.points) >= 3
        # Streamline should primarily extend along Z
        z_span = sl.points[:, 2].max() - sl.points[:, 2].min()
        assert z_span > 3.0

    def test_fa_threshold_stops_tracking(self):
        v1, fa = self._setup_straight_field()
        fa[:, :, :4] = 0.05  # low FA barrier
        seed = np.array([6.0, 6.0, 6.0])
        sl = track_streamline_bidirectional(
            seed=seed, v1_field=v1, fa_map=fa,
            step_size=0.5, fa_threshold=0.15, max_angle_deg=40.0,
            max_steps=200, method="rk4"
        )
        assert sl is not None
        # Should not cross into low-FA region (z < 4)
        assert sl.points[:, 2].min() >= 3.5

    def test_ensemble_count(self):
        v1, fa = self._setup_straight_field()
        seeds = [(6.0, 6.0, 6.0), (5.0, 5.0, 6.0), (7.0, 7.0, 5.0)]
        tg = track_ensemble(
            v1_field=v1, fa_map=fa, seeds=seeds,
            step_size=0.5, fa_threshold=0.2, min_length=2.0,
            method="rk4", seed=0
        )
        assert isinstance(tg, Tractogram)
        assert tg.count >= 2
        assert tg.mean_length > 0

    def test_euler_method_also_works(self):
        v1, fa = self._setup_straight_field()
        seed = np.array([6.0, 6.0, 6.0])
        sl = track_streamline_bidirectional(
            seed=seed, v1_field=v1, fa_map=fa,
            step_size=0.5, method="euler", max_steps=50
        )
        assert sl is not None
        assert sl.length > 1.0

    def test_low_fa_seed_returns_none(self):
        v1, fa = self._setup_straight_field()
        fa[:] = 0.05
        sl = track_streamline_bidirectional(
            seed=np.array([6.0, 6.0, 6.0]),
            v1_field=v1, fa_map=fa, fa_threshold=0.15
        )
        assert sl is None


# ─────────────────────────────────────────────────────────────────────────────
# Connectome
# ─────────────────────────────────────────────────────────────────────────────

class TestConnectome:
    def test_atlas_shape_and_labels(self):
        atlas = build_parcellation_atlas(grid_size=(16, 16, 10), n_regions=6)
        assert atlas.shape == (16, 16, 10)
        labels = set(np.unique(atlas).tolist())
        assert labels.issubset(set(range(7)))  # 0..6
        assert max(labels) == 6

    def test_extract_connectome_metrics(self):
        # Create synthetic streamlines connecting known ROIs
        atlas = build_parcellation_atlas(grid_size=(20, 20, 12), n_regions=6)
        # Manual streamline from ROI 1 region to ROI 2 region
        pts1 = np.array([[5.0, 10.0, 9.0], [8.0, 10.0, 9.0], [12.0, 10.0, 9.0]])
        pts2 = np.array([[5.0, 5.0, 3.0], [10.0, 5.0, 3.0], [15.0, 5.0, 3.0]])
        sl1 = Streamline(points=pts1, length=7.0, fa_profile=np.array([0.5, 0.6, 0.5]))
        sl2 = Streamline(points=pts2, length=10.0, fa_profile=np.array([0.4, 0.5, 0.4]))
        tg = Tractogram(streamlines=[sl1, sl2], count=2, mean_length=8.5)

        graph = extract_connectome(tg, atlas, n_regions=6)
        assert isinstance(graph, ConnectomeGraph)
        assert graph.adjacency.shape == (6, 6)
        assert graph.adjacency.T == pytest.approx(graph.adjacency)  # symmetric
        assert 0.0 <= graph.density <= 1.0
        assert 0.0 <= graph.clustering_coefficient <= 1.0
        assert graph.global_efficiency >= 0.0
        assert graph.node_degrees.shape == (6,)

    def test_empty_tractogram_zero_density(self):
        atlas = build_parcellation_atlas(grid_size=(8, 8, 6), n_regions=4)
        tg = Tractogram(streamlines=[], count=0, mean_length=0.0)
        graph = extract_connectome(tg, atlas, n_regions=4)
        assert graph.density == 0.0
        assert np.allclose(graph.adjacency, 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# End-to-End Integration
# ─────────────────────────────────────────────────────────────────────────────

class TestEndToEnd:
    def test_full_pipeline(self):
        """Phantom → DTI → Tractography → Connectome must complete without error."""
        ph = create_curved_bundle_phantom(
            grid_size=(18, 18, 12), snr=50.0, n_directions=16, seed=99
        )
        dti = fit_dti_volume(ph.dwi, ph.bvecs, ph.bvals, mask=ph.mask)

        # Seeds in high-FA voxels
        seeds_idx = np.argwhere((dti.fa > 0.25) & ph.mask)
        rng = np.random.default_rng(0)
        n = min(40, len(seeds_idx))
        if n == 0:
            pytest.skip("No high-FA seeds in phantom")
        chosen = seeds_idx[rng.choice(len(seeds_idx), size=n, replace=False)]
        seeds = [tuple(float(c) for c in s) for s in chosen]

        v1 = dti.eigenvectors[..., :, 0]
        tg = track_ensemble(
            v1_field=v1, fa_map=dti.fa, seeds=seeds,
            step_size=0.5, fa_threshold=0.12, min_length=3.0,
            max_steps=200, method="rk4", seed=1
        )
        assert tg.count >= 1, "Expected at least one streamline"

        atlas = build_parcellation_atlas(grid_size=ph.dwi.shape[:3], n_regions=6)
        graph = extract_connectome(tg, atlas, n_regions=6)
        assert graph.adjacency.shape == (6, 6)
        # FA should be reasonable
        assert 0.0 <= float(np.mean(dti.fa[ph.mask])) <= 1.0

    def test_deterministic_reproducibility(self):
        ph1 = create_curved_bundle_phantom(grid_size=(12, 12, 8), seed=123, n_directions=10)
        ph2 = create_curved_bundle_phantom(grid_size=(12, 12, 8), seed=123, n_directions=10)
        np.testing.assert_allclose(ph1.dwi, ph2.dwi)
