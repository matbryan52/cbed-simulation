import os
import io
import re
from typing import NamedTuple
import pathlib
import pandas as pd
import numpy as np
import pytest
from numpy.testing import assert_allclose

from cbed_simulation.crystal_orientation import (
    OrientedPhase, ExperimentInformation, IndexedPeaks, EulerAngles
)


class ASTARTemplate(NamedTuple):
    CX: int
    CY: int
    Radius: int


ROOT_PATH = pathlib.Path(__file__).parent


def load_ref(data_path: os.PathLike, scale=1 / 71):
    with pathlib.Path(data_path).open("r") as fp:
        lines = fp.readlines()

    cif = lines[0]
    cif = cif.removesuffix('\n')
    euler = EulerAngles(*map(float, lines[1].replace(" ", "").strip().split(",")))
    offset = ASTARTemplate(
        *map(int, re.sub(r"\s+", ",", lines[5].lstrip(" ").rstrip("\n")).split(","))
    )
    lines = lines[12:]
    lines = [re.sub(r"\s+", ",", line.lstrip(" ").rstrip("\n")) for line in lines]
    lines = "\n".join(lines)
    df = pd.read_csv(io.StringIO(lines))

    # NOTE need to apply a flip-X i.e. multiply -1, here
    ref_x = (df['bmx'] - offset.CX) * scale
    ref_y = (df['bmy'] - offset.CY) * scale
    ref_peaks = IndexedPeaks(
        pos_000=complex(offset.CX, offset.CY),
        offsets=ref_x + ref_y * 1j,
        hkls=df[['h', 'k', 'l']].to_numpy(),
        weights=np.ones_like(ref_x),
    )
    return ref_peaks, euler


@pytest.mark.parametrize(
        "cif_path, ref_path",
        (
            (ROOT_PATH / "Si.cif", ROOT_PATH / "ASTAR_Si_Euler0.txt"),
            (ROOT_PATH / "Si.cif", ROOT_PATH / "ASTAR_Si_Euler1.txt"),
            (ROOT_PATH / "Si.cif", ROOT_PATH / "ASTAR_Si_Euler2.txt"),
            (ROOT_PATH / "Si.cif", ROOT_PATH / "ASTAR_Si_Euler3.txt"),
            # (ROOT_PATH / "GaN.cif", ROOT_PATH / "ASTAR_GaN_Euler0.txt"),
            # (ROOT_PATH / "GaN.cif", ROOT_PATH / "ASTAR_GaN_Euler1.txt"),
            # (ROOT_PATH / "GaN.cif", ROOT_PATH / "ASTAR_GaN_Euler2.txt"),
            # (ROOT_PATH / "GaN.cif", ROOT_PATH / "ASTAR_GaN_Euler3.txt"),
        )
    )
def test_ref_comparison(cif_path: os.PathLike, ref_path: os.PathLike):
    ref_peaks, euler = load_ref(ref_path)
    ref_hkls = set(tuple(hkl) for hkl in ref_peaks.hkls)

    experiment = ExperimentInformation(
        frame_shape=(512, 512),
        transmitted_centre_px=complex(256, 256),
        radius_px=12,
        pattern_scale_factor=119.,  # pixels / Å-1
    )

    phase = OrientedPhase.from_cif(
        cif_path=cif_path,
        orientation=euler,
    )
    sim_peaks = phase.peak_positions(
        experiment, max_excitation_error=0.05,
    )
    sim_hkls = set(tuple(hkl) for hkl in sim_peaks.hkls)

    # Compare ref & sim
    common_hkls = sim_hkls.intersection(ref_hkls)
    ref_pos = tuple(ref_peaks.spot_position(hkl, centre_zero=True) for hkl in common_hkls)
    sim_pos = tuple(sim_peaks.spot_position(hkl, centre_zero=True) for hkl in common_hkls)

    assert_allclose(
        actual=sim_pos,
        desired=ref_pos,
        atol=2.5e-2,
    )
