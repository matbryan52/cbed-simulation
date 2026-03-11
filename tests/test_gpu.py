import pathlib
import pytest

from cbed_simulation.crystal_orientation import (
    ExperimentInformation, OrientedPhase,
)
from cbed_simulation.frame_builder import FrameParameters

try:
    import cupy as cp
except ImportError:
    pytest.skip("No cupy available, skipping GPU tests")


ROOT_PATH = pathlib.Path(__file__).parent


def test_gpu_frame_gen():
    backend = "cupy"
    experiment = ExperimentInformation(
        frame_shape=(512, 512),
        transmitted_centre_px=complex(240, 281),
        radius_px=12,
        pattern_scale_factor=200.,  # pixels / Å-1
    )
    phase = OrientedPhase.from_cif(
        cif_path=ROOT_PATH / "Si.cif",
        zone_axis=(1, 1, 0),
    )
    sim_peaks = phase.peak_positions(
        experiment,
        backend=backend,
    )
    frame_params = FrameParameters()
    frame_ref = phase.synthetic(
        experiment,
        sim_peaks,
        frame_params=frame_params,
        backend=backend,
    )
    assert frame_ref.device == cp.cuda.Device(0)
