import pytest
import pathlib
import numpy as np
import libertem.api as lt
from cbed_simulation.udf import CBEDSimUDF
from cbed_simulation.crystal_orientation import ExperimentInformation, OrientedPhase, IndexedPeaks
from cbed_simulation.frame_builder import FrameParameters


@pytest.fixture(scope="module")
def ctx():
    return lt.Context.make_with("inline")


def test_udf_gen(ctx: lt.Context, tmp_path: pathlib.Path):
    experiment = ExperimentInformation(
        frame_shape=(256, 256),
        transmitted_centre_px=complex(128, 128),
        radius_px=12,
        pattern_scale_factor=200.,  # pixels / Å-1
    )
    frame_params = FrameParameters()
    phase = OrientedPhase.from_cif("Si.cif")

    data = np.asarray((0, 0, 0), dtype=np.float32).reshape(1, 3)
    ds = ctx.load("memory", data=data, sig_shape=(3,))
    out_path = tmp_path / "frames.npy"
    udf = CBEDSimUDF(
        filename=out_path,
        experiment=experiment,
        phase=phase,
        frame_params=frame_params,
    )
    res = ctx.run_udf(ds, udf)
    assert out_path.is_file()
    frames = np.load(out_path)
    assert frames.shape == (1, *experiment.frame_shape)
    peaks: IndexedPeaks = res["peak_positions"].data[0]
    assert peaks.size == 5
