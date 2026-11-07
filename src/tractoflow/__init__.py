"""
TractoFlow — Diffusion Tensor Imaging (DTI) & Deterministic/Probabilistic
Streamline Fiber Tractography Engine for Brain Connectomics.

Public API:
  - fit_dti_volume, DTIFitResult, construct_design_matrix
  - create_curved_bundle_phantom, generate_gradient_scheme, SyntheticDWIDataset
  - track_ensemble, track_streamline_bidirectional, Tractogram, Streamline
  - extract_connectome, build_parcellation_atlas, ConnectomeGraph
"""

from .dti import (
    DTIFitResult,
    construct_design_matrix,
    fit_dti_volume,
)
from .phantom import (
    SyntheticDWIDataset,
    create_curved_bundle_phantom,
    generate_gradient_scheme,
)
from .tractography import (
    Streamline,
    Tractogram,
    track_ensemble,
    track_streamline_bidirectional,
    rk4_step,
    trilinear_interpolate_vector,
)
from .connectome import (
    ConnectomeGraph,
    build_parcellation_atlas,
    extract_connectome,
)

__version__ = "0.1.0"
__all__ = [
    "DTIFitResult",
    "construct_design_matrix",
    "fit_dti_volume",
    "SyntheticDWIDataset",
    "create_curved_bundle_phantom",
    "generate_gradient_scheme",
    "Streamline",
    "Tractogram",
    "track_ensemble",
    "track_streamline_bidirectional",
    "rk4_step",
    "trilinear_interpolate_vector",
    "ConnectomeGraph",
    "build_parcellation_atlas",
    "extract_connectome",
    "__version__",
]
