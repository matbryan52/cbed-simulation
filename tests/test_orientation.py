import pathlib
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal

import py4DSTEM
from py4DSTEM.process.diffraction.crystal import Crystal
from py4DSTEM.braggvectors.braggvectors import BraggVectors
from emdfile import PointListArray, PointList
from scipy.spatial.transform import Rotation as RotationSP

from cbed_simulation.crystal_orientation import (
    EulerAngles, OrientedPhase, ExperimentInformation
)
from cbed_simulation.frame_builder import FrameParameters


ROOT_PATH = pathlib.Path(__file__).parent


def reduce_orientation(crystal: Crystal, matrix: np.ndarray):
    family = np.zeros_like(matrix)
    for a0 in range(3):
        in_range = np.all(
            np.sum(
                crystal.symmetry_reduction
                * matrix[:, a0][None, :, None],
                axis=1,
            )
            >= 0,
            axis=1,
        )

        family[:, a0] = (
            crystal.symmetry_operators[np.argmax(in_range)]
            @ matrix[:, a0]
        )
    return family


@pytest.mark.parametrize(
        "cif_path", (ROOT_PATH / "Si.cif",)
)
def test_001_no_rotation(cif_path):
    phase = OrientedPhase.from_cif(
        cif_path=cif_path,
        zone_axis=(0, 0, 1),
    )
    assert_allclose(phase.orientation.to_matrix().squeeze(), np.eye(3))


@pytest.mark.parametrize(
        "cif_path, za, euler",
        (
            (ROOT_PATH / "Si.cif", (1, 1, 0), (315, 90, 45)),
            (ROOT_PATH / "Si.cif", (1, 1, 1), (315, 54.74, 45)),
            (ROOT_PATH / "Si.cif", (2, 0, 1), (270, 63.43, 90)),
            (ROOT_PATH / "Si.cif", (0, 3, 0), (0, 90, 0)),
            (ROOT_PATH / "Si.cif", (0, 3, 2), (0, 56.3, 0)),
            (ROOT_PATH / "GaN.cif", (1, 1, 0), (330, 90, 30)),
            # (ROOT_PATH / "GaN.cif", (2, 0, 1), (270, 50.83, 90)),
        )
)
def test_za_euler_equal(cif_path, za, euler):
    # Note first Euler angle is in-plane rotation so is arbitrary
    # for a ZA orientation when comparing to ASTAR
    # GaN rotations seem broken except for the simplest ?
    phase = OrientedPhase.from_cif(
        cif_path=cif_path,
        zone_axis=za,
    )
    assert_allclose(
        phase.orientation.to_euler(degrees=True).squeeze(),
        euler,
        rtol=0,
        atol=0.1,
    )


@pytest.mark.parametrize(
        "cif_path", (
            ROOT_PATH / "Si.cif",
            ROOT_PATH / "GaN.cif",
        )
)
@pytest.mark.parametrize(
        "in_plane", (26, -73,)
)
@pytest.mark.parametrize(
        "za", ((0, 0, 1), (1, 1, 0))
)
@pytest.mark.parametrize(
        "bloch", (False, True)
)
def test_in_plane_rotation(cif_path, in_plane, za, bloch):
    phase = OrientedPhase.from_cif(
        cif_path=cif_path,
        zone_axis=za,
    )
    phase_rot = phase.with_rot(
        zone_axis=za,
        in_plane_rot=in_plane,
    )
    experiment = ExperimentInformation(
        frame_shape=(256, 256),
        transmitted_centre_px=complex(128, 128),
        radius_px=12,
        pattern_scale_factor=119.
    )
    peaks = phase.peak_positions(experiment, dynamic_diff=bloch)
    peaks_rot = phase_rot.peak_positions(experiment, dynamic_diff=bloch)

    hkl = tuple(peaks.hkls[5])
    assert hkl != (0, 0, 0)
    pos = peaks.spot_position(hkl, centre_zero=True)
    pos_rot = peaks_rot.spot_position(hkl, centre_zero=True)
    assert_allclose(
        np.rad2deg(np.angle(pos_rot) - np.angle(pos)),
        in_plane,
    )


@pytest.mark.parametrize(
    "cif_path", (
        ROOT_PATH / "Si.cif",
        ROOT_PATH / "GaN.cif",
    )
)
@pytest.mark.parametrize(
    "orientation", (
        dict(zone_axis=None),
        dict(zone_axis=(1, 1, 0)),
        dict(zone_axis=(2, 0, 1)),
        dict(orientation=(30, 10, 70)),
    )
)
def test_kinematic_dynamic_equivalent(cif_path, orientation):
    phase = OrientedPhase.from_cif(
        cif_path=cif_path,
        **orientation,
    )
    experiment = ExperimentInformation(
        frame_shape=(256, 256),
        transmitted_centre_px=complex(128, 128),
        radius_px=12,
        pattern_scale_factor=119.,  # pixels / Å-1
    )
    peaks_kinematic = phase.peak_positions(experiment, dynamic_diff=False)
    peaks_dynamic = phase.peak_positions(experiment, dynamic_diff=True)

    # make kinematic ⊆ dynamic
    peaks_kinematic_filt, _ = peaks_kinematic.match_peaks(peaks_dynamic)
    distances = (
        peaks_kinematic_filt.offsets[:, np.newaxis]
        - peaks_dynamic.offsets[np.newaxis, :]
    )
    distances = np.abs(distances)
    matches = distances.argmin(axis=1)
    min_distances = distances.min(axis=1)

    # import matplotlib.pyplot as plt
    # from cbed_simulation.utils import overlay_peaks
    # _, ax = overlay_peaks(
    #     peaks_dynamic, peaks_kinematic_filt, names=("Dynamic", "Kinematic")
    # )
    # ax.set_title(f"{cif_path.stem} - {orientation}")
    # ax.legend()
    # plt.savefig("out.png")

    # matches are very close
    assert_allclose(min_distances, 0., atol=1e-5, rtol=0.)
    # no double matches
    assert matches.size == np.unique(matches).size
    # matches have equivalent hkl values
    assert_equal(peaks_kinematic_filt.hkls, peaks_dynamic.hkls[matches])


@pytest.fixture(scope="module")
def si_plan():
    cif_path = ROOT_PATH / "Si.cif"
    # Load crystal
    crystal = py4DSTEM.process.diffraction.Crystal.from_CIF(
        CIF=cif_path,
        conventional_standard_structure=True,
    )

    # Create orientation plan
    k_max = 2.0
    angle_step_in_plane = angle_step_zone_axis = 3.
    crystal.calculate_structure_factors(k_max=k_max)
    crystal.orientation_plan(
        zone_axis_range='full',
        accel_voltage=200e3,
        precession_angle_degrees=1.,
        angle_step_zone_axis=angle_step_zone_axis,
        angle_step_in_plane=angle_step_in_plane,
        progress_bar=False,
    )
    yield cif_path, crystal


def wrap_bunge_deg(eulers_deg):
    """Wrap Bunge Euler angles: phi1,phi2 in [0,360), Phi in [0,180]."""
    phi1, Phi, phi2 = map(float, eulers_deg)
    phi1 %= 360.0
    phi2 %= 360.0

    Phi %= 360.0
    if Phi > 180.0:
        Phi = 360.0 - Phi
        phi1 = (phi1 + 180.0) % 360.0
        phi2 = (phi2 + 180.0) % 360.0

    return np.array([phi1, Phi, phi2], float)


@pytest.mark.parametrize(
        "plan, euler",
        (
            ("si_plan", (75, 10, 10)),
            ("si_plan", (10, 15, 0)),
            ("si_plan", (0, 60, 10)),
        )
)
def test_py4DSTEM_orientation(plan: str, euler: EulerAngles, request):
    cif_path, crystal = request.getfixturevalue(plan)
    cif_path: pathlib.Path
    crystal: Crystal

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
        experiment, max_excitation_error=0.05, dynamic_diff=True
    )
    sim_peaks_px = sim_peaks.to_pixels(experiment)

    dt = [("qx", np.float64), ("qy", np.float64), ("intensity", np.float64)]
    pointlist = PointListArray(
        dtype=dt,
        shape=(1, 1),
        name="_v_uncal",
    )
    points = [
        tuple(a)
        for a in
        zip(sim_peaks_px.peaks.imag, sim_peaks_px.peaks.real, sim_peaks_px.weights)
    ]
    pointlist[0, 0] = PointList(
        np.asarray(points, dtype=dt),
    )
    braggpeaks = BraggVectors(
        (1, 1), experiment.frame_shape,
    )
    braggpeaks.set_raw_vectors(pointlist)
    braggpeaks.calibration.set_origin(
        (sim_peaks_px.pos_000.real, sim_peaks_px.pos_000.imag),
    )
    braggpeaks.calibration.set_Q_pixel_size(1 / experiment.pattern_scale_factor)
    braggpeaks.calibration.set_Q_pixel_units('A^-1')
    braggpeaks.setcal()

    # Create and plot orientation map
    orientation_map = crystal.match_orientations(
        braggpeaks,
        progress_bar=False,
    )

    # Extract and transform orientation
    or_matrix = orientation_map.matrix[0, 0, 0]

    # Convert to ASTAR / cbed_simulation convention
    angles = RotationSP.from_matrix(or_matrix).as_euler("zxz")
    angles[0] -= np.pi/2
    angles *= -1
    angles = np.rad2deg(angles)
    angles = wrap_bunge_deg(angles)

    # Simulate peak positions from the inferred orientation
    or_phase = OrientedPhase.from_cif(cif_path=cif_path, orientation=angles)
    or_sim_peaks = or_phase.peak_positions(
        experiment, max_excitation_error=0.05, dynamic_diff=True,
    )

    peak_distances = np.abs(or_sim_peaks.offsets[:, np.newaxis] - sim_peaks.offsets[np.newaxis, :])
    peak_matches = peak_distances < 5e-2
    or_matches_peaks = peak_matches.any(axis=1)
    sim_matches_peaks = peak_matches.any(axis=0)

    if False:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
        ax.axhline(alpha=0.2, color="k")
        ax.axvline(alpha=0.2, color="k")
        ax.plot(sim_peaks.offsets[sim_matches_peaks].real, sim_peaks.offsets[sim_matches_peaks].imag, 'ko', label="Input-EA", alpha=0.5)
        ax.plot(or_sim_peaks.offsets[or_matches_peaks].real, or_sim_peaks.offsets[or_matches_peaks].imag, 'rx', label="py4DSTEM-EA")
        ax.plot(sim_peaks.offsets[~sim_matches_peaks].real, sim_peaks.offsets[~sim_matches_peaks].imag, 'k.', label="Input-EA-extra", alpha=0.33)
        ax.plot(or_sim_peaks.offsets[~or_matches_peaks].real, or_sim_peaks.offsets[~or_matches_peaks].imag, 'r.', label="py4DSTEM-EA-extra", alpha=0.33)
        ax.yaxis.set_inverted(True)
        ax.axis("equal")
        ax.legend()
        ax.set_title(f"In: {euler}, Out: {tuple(np.round(angles, decimals=1))}")
        fig.tight_layout()
        plt.savefig(ROOT_PATH / f"out_{cif_path.stem}_{euler}.png")
        fp = FrameParameters(intensity_from_radius=True)
        frame = or_phase.synthetic(experiment, sim_peaks, frame_params=fp)
        plt.imsave(ROOT_PATH / "out_frame.png", frame)

    match_frac = 0.5
    assert or_matches_peaks.sum() > match_frac * or_matches_peaks.size
    assert sim_matches_peaks.sum() > match_frac * sim_matches_peaks.size
