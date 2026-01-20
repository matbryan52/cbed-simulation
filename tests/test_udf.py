import pytest
import pathlib
import numpy as np
import libertem.api as lt
from cbed_simulation.udf import CBEDSimUDF, build_udf_ds
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

    out_path = tmp_path / "frames.npy"
    udf, ds = build_udf_ds(
        out_path,
        (1,),
        ctx,
        phase,
        experiment,
        frame_parameters=frame_params,
    )
    res = ctx.run_udf(ds, udf)
    assert out_path.is_file()
    frames = np.load(out_path)
    assert frames.shape == (1, *experiment.frame_shape)
    peaks: IndexedPeaks = res["peak_positions"].data[0]
    assert peaks.size == 5
