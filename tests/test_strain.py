import pathlib
import pytest
import numpy as np
from numpy.testing import assert_allclose

import py4DSTEM
from py4DSTEM.braggvectors.braggvectors import BraggVectors
from emdfile import PointListArray, PointList

from cbed_simulation.crystal_orientation import ExperimentInformation, OrientedPhase
from cbed_simulation.template_from_image import template_from_vacuum, shift_probe
from cbed_simulation.frame_builder import FrameParameters

from cbed_simulation.strain_decomposition import compute_strain_large_def

ROOT_PATH = pathlib.Path(__file__).parent


def save_frames(frame_ref, frame_strained, savedir=pathlib.Path(".")):
    import matplotlib.pyplot as plt
    import imageio as iio

    plt.imsave(savedir / "out_frame.png", frame_ref)
    plt.imsave(savedir / "out_frame_strained.png", frame_strained)
    im1 = iio.v3.imread(savedir / "out_frame.png")
    im2 = iio.v3.imread(savedir / "out_frame_strained.png")
    iio.v3.imwrite(savedir / "out.gif", (im1, im2), loop=0, duration=500)
    # import tifffile
    # tifffile.imwrite(
    #     savedir / "out_frame.tiff",
    #     frame_ref.astype(np.int32),
    #     photometric='minisblack',
    # )


def ref_strained_crystal(cif_path, strain_val):
    experiment = ExperimentInformation(
        frame_shape=(512, 512),
        transmitted_centre_px=complex(240, 281),
        radius_px=12,
        pattern_scale_factor=200.,  # pixels / Å-1
    )
    phase = OrientedPhase.from_cif(
        cif_path=cif_path,
        zone_axis=(0, 0, 1),
    )
    sim_peaks_ref = phase.peak_positions(experiment)
    sim_peaks_strained = phase.peak_positions(experiment, stretch_abc=strain_val)

    g1_hkl = (4, 0, 0)
    g2_hkl = (0, 4, 0)
    return (sim_peaks_ref, sim_peaks_strained), strain_val, (g1_hkl, g2_hkl), (phase, experiment)


@pytest.mark.parametrize(
        "cif_path, strain_val",
        (
            (ROOT_PATH / "Si.cif", (1.01, 1., 1.)),
            (ROOT_PATH / "Si.cif", (1., 1.03, 1.)),
            (ROOT_PATH / "Si.cif", (1.03, 1., 1.)),
            (ROOT_PATH / "Si.cif", (0.98, 1., 1.)),
        )
)
def test_gem_ed_2D_strain_equations(cif_path, strain_val):
    (
        (sim_peaks_ref, sim_peaks_strained),
        strain_val,
        (g1_hkl, g2_hkl),
        (_, experiment),
    ) = ref_strained_crystal(cif_path, strain_val)

    sim_pos_ref_px = sim_peaks_ref.to_pixels(experiment)
    sim_pos_strained_px = sim_peaks_strained.to_pixels(experiment)

    strain_res = compute_strain_large_def(
        sim_pos_strained_px.spot_position(g1_hkl, centre_zero=True),
        sim_pos_strained_px.spot_position(g2_hkl, centre_zero=True),
        sim_pos_ref_px.spot_position(g1_hkl, centre_zero=True),
        sim_pos_ref_px.spot_position(g2_hkl, centre_zero=True),
    ).to_vector(
        sim_pos_ref_px.spot_position(g1_hkl, centre_zero=True)
    )
    assert_allclose(strain_res.e_xx, strain_val[0] - 1., rtol=0., atol=1e-4)
    assert_allclose(strain_res.e_xy, 0., rtol=0., atol=1e-4)
    assert_allclose(strain_res.e_yy, strain_val[1] - 1., rtol=0., atol=1e-4)
    assert_allclose(strain_res.theta, 0., rtol=0., atol=1e-4)


POINTLIST_DT = [("qx", np.float64), ("qy", np.float64), ("intensity", np.float64)]


def to_pointlist(peaks):
    points_ref = [
        tuple(a)
        for a in
        zip(
            peaks.peaks.imag,
            peaks.peaks.real,
            peaks.weights,
        )
    ]
    return PointList(
        np.asarray(points_ref, dtype=POINTLIST_DT),
    )


def strainmap_to_tensors(strainmap):
    e_xx = strainmap.data[0].squeeze()
    e_yy = strainmap.data[1].squeeze()
    e_xy = strainmap.data[2].squeeze()
    theta = strainmap.data[3].squeeze()

    tensor = np.stack([
        np.stack([e_xx, e_xy], axis=-1),
        np.stack([e_xy, e_yy], axis=-1),
    ], axis=1)
    return tensor, theta


def strainval_to_target(strain_val):
    target_tensor = np.zeros((2, 2))
    target_tensor[0, 0] = strain_val[1] - 1.
    target_tensor[1, 1] = strain_val[0] - 1.
    return target_tensor


@pytest.mark.parametrize(
        "cif_path, strain_val",
        (
            (ROOT_PATH / "Si.cif", (1.01, 1., 1.)),
            (ROOT_PATH / "Si.cif", (1., 1.03, 1.)),
            (ROOT_PATH / "Si.cif", (1.03, 1., 1.)),
            (ROOT_PATH / "Si.cif", (0.98, 1., 1.)),
        )
)
def test_py4DSTEM_2D_strain_equations(cif_path, strain_val):
    (
        (sim_peaks_ref, sim_peaks_strained),
        strain_val,
        (g1_hkl, g2_hkl),
        (_, experiment),
    ) = ref_strained_crystal(cif_path, strain_val)

    sim_pos_ref_px = sim_peaks_ref.to_pixels(experiment, clip=True)
    sim_pos_strained_px = sim_peaks_strained.to_pixels(experiment, clip=True)

    pointlist = PointListArray(
        dtype=POINTLIST_DT,
        shape=(1, 2),
        name="_v_uncal",
    )
    pointlist[0, 0] = to_pointlist(sim_pos_ref_px)
    pointlist[0, 1] = to_pointlist(sim_pos_strained_px)

    braggpeaks = BraggVectors(
        (1, 2), experiment.frame_shape,
    )
    braggpeaks.set_raw_vectors(pointlist)
    braggpeaks.calibration.set_origin(
        (sim_pos_ref_px.pos_000.imag, sim_pos_ref_px.pos_000.real),
    )
    braggpeaks.setcal(center=True)

    strainmap = py4DSTEM.StrainMap(
        braggvectors=braggpeaks
    )
    strainmap.choose_basis_vectors(
        index_g1=sim_pos_ref_px.spot_index(g1_hkl),
        index_g2=sim_pos_ref_px.spot_index(g2_hkl),
        maxNumPeaks=16,
    )
    strainmap.fit_basis_vectors(
        max_peak_spacing=2,
    )
    ref_roi = np.arange(1, -1, -1).astype(bool).reshape(1, 2)
    g_ref = strainmap.get_reference_g1g2(ref_roi)
    strainmap.get_strain(g_ref)

    (ref_tensor, strain_tensor), theta = strainmap_to_tensors(strainmap)
    target_tensor = strainval_to_target(strain_val)

    tol = dict(rtol=0., atol=1e-3)
    assert_allclose(ref_tensor, np.zeros((2, 2)), **tol)
    assert_allclose(strain_tensor, target_tensor, **tol)
    assert_allclose(theta, np.zeros((2,)), **tol)


@pytest.mark.parametrize(
        "cif_path, strain_val",
        (
            (ROOT_PATH / "Si.cif", (1., 1.03, 1.)),
            (ROOT_PATH / "Si.cif", (1.01, 1., 1.)),
            (ROOT_PATH / "Si.cif", (1.03, 1., 1.)),
            (ROOT_PATH / "Si.cif", (0.98, 1., 1.)),
        )
)
def test_py4DSTEM_template_match(cif_path, strain_val):
    (
        (sim_peaks_ref, sim_peaks_strained),
        strain_val,
        _,
        (phase, experiment),
    ) = ref_strained_crystal(cif_path, strain_val)

    frame_params = FrameParameters(
        textured=False,
        disk_blur_sigma=0.75,
        inelastic_scatter_sigma=0.,
        additive_noise_scale=0.,
        intensity_from_radius=True,
    )

    frame_ref = phase.synthetic(experiment, sim_peaks_ref, frame_params=frame_params)
    frame_strained = phase.synthetic(experiment, sim_peaks_strained, frame_params=frame_params)

    frames = np.stack((frame_ref, frame_strained), axis=0)
    cube = py4DSTEM.DataCube(frames[np.newaxis, ...])

    cyx = (experiment.transmitted_centre_px.imag, experiment.transmitted_centre_px.real)
    template = template_from_vacuum(frame_ref, cyx, experiment.radius_px)
    template_shifted = shift_probe(
        template, cyx, shifted="bilinear"
    )

    braggpeaks = cube.find_Bragg_disks(
        template=template_shifted,
        corrPower=1.0,
        sigma=0,
        edgeBoundary=16,
        minRelativeIntensity=0.025,
        minAbsoluteIntensity=1,
        minPeakSpacing=8,
        subpixel='poly',
        upsample_factor=20,
        maxNumPeaks=32,
    )

    sim_pos_px = sim_peaks_ref.to_pixels(experiment)
    detected_points = braggpeaks.raw[0, 0].qy + braggpeaks.raw[0, 0].qx * 1j
    distances = detected_points[:, np.newaxis] - sim_pos_px.peaks[np.newaxis, :]
    distances = np.abs(distances)
    min_distances = distances.min(axis=1)
    assert_allclose(min_distances, 0., rtol=0., atol=0.1)


@pytest.mark.parametrize(
        "cif_path, strain_val",
        (
            (ROOT_PATH / "Si.cif", (1., 1.03, 1.)),
            (ROOT_PATH / "Si.cif", (1.03, 1., 1.)),
            (ROOT_PATH / "Si.cif", (0.98, 1., 1.)),
        )
)
def test_py4DSTEM_2D_strain_framegen(cif_path, strain_val):
    (
        (sim_peaks_ref, sim_peaks_strained),
        strain_val,
        (g1_hkl, g2_hkl),
        (phase, experiment),
    ) = ref_strained_crystal(cif_path, strain_val)

    frame_params = FrameParameters(
        textured=False,
        disk_blur_sigma=0.,
        inelastic_scatter_sigma=0.,
        additive_noise_scale=0.,
        intensity_from_radius=True,
    )

    frame_ref = phase.synthetic(experiment, sim_peaks_ref, frame_params=frame_params)
    frame_strained = phase.synthetic(experiment, sim_peaks_strained, frame_params=frame_params)

    frames = np.stack((frame_ref, frame_strained), axis=0)
    cube = py4DSTEM.DataCube(frames[np.newaxis, ...])

    cyx = (experiment.transmitted_centre_px.imag, experiment.transmitted_centre_px.real)
    template = template_from_vacuum(frame_ref, cyx, experiment.radius_px)
    template_shifted = shift_probe(
        template, cyx, shifted="bilinear"
    )

    braggpeaks = cube.find_Bragg_disks(
        template=template_shifted,
        corrPower=1.0,
        sigma=0,
        edgeBoundary=16,
        minRelativeIntensity=0.025,
        minAbsoluteIntensity=1,
        minPeakSpacing=8,
        subpixel='poly',
        upsample_factor=20,
        maxNumPeaks=32,
    )

    sim_pos_px = sim_peaks_ref.to_pixels(experiment)
    detected_points = braggpeaks.raw[0, 0].qy + braggpeaks.raw[0, 0].qx * 1j
    distances = detected_points[:, np.newaxis] - sim_pos_px.peaks[np.newaxis, :]
    distances = np.abs(distances)
    min_distances = distances.min(axis=1)
    assert_allclose(min_distances, 0., rtol=0., atol=0.1)

    idx_g1 = sim_pos_px.spot_index(g1_hkl)
    idx_g2 = sim_pos_px.spot_index(g2_hkl)
    min_dist_idxs = distances.argmin(axis=1)
    idx_g1 = np.argwhere(min_dist_idxs == idx_g1).item()
    idx_g2 = np.argwhere(min_dist_idxs == idx_g2).item()

    braggpeaks.calibration.set_origin(
        (experiment.transmitted_centre_px.imag, experiment.transmitted_centre_px.real)
    )
    braggpeaks.setcal(center=True)

    ref_roi = np.arange(1, -1, -1).astype(bool).reshape(1, 2)
    strainmap = py4DSTEM.StrainMap(
        braggvectors=braggpeaks
    )
    strainmap.choose_basis_vectors(
        index_g1=idx_g1,
        index_g2=idx_g2,
        maxNumPeaks=32,
    )
    strainmap.fit_basis_vectors(
        max_peak_spacing=2,
    )
    g_ref = strainmap.get_reference_g1g2(ref_roi)
    strainmap.get_strain(g_ref)

    (ref_tensor, strain_tensor), theta = strainmap_to_tensors(strainmap)
    target_tensor = strainval_to_target(strain_val)

    tol = dict(rtol=0., atol=1e-3)
    assert_allclose(ref_tensor, np.zeros((2, 2)), **tol)
    assert_allclose(strain_tensor, target_tensor, **tol)
    assert_allclose(theta, np.zeros((2,)), **tol)
