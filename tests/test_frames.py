import pathlib
import pytest
import numpy as np  # noqa
from numpy.testing import assert_allclose
from scipy.ndimage import center_of_mass
from scipy.signal import peak_widths

from cbed_simulation.crystal_orientation import ExperimentInformation, IndexedPeaks
from cbed_simulation.frame_builder import FrameParameters, build_frame
from cbed_simulation.utils import to_array


ROOT_PATH = pathlib.Path(__file__).parent


@pytest.mark.parametrize("radius", (5, 6, 12, 17))
@pytest.mark.parametrize("blur_sigma", (0, 0.5, 1.))
def test_fwhm(radius, blur_sigma):
    experiment = ExperimentInformation(
        frame_shape=(256, 256),
        pattern_scale_factor=200.,  # pixels / Å-1
        radius_px=radius,
    )
    frame_params = FrameParameters(
        intensity_from_radius=True,
        textured=False,
        poisson_frame=False,
        disk_blur_sigma=blur_sigma,
        inelastic_scatter_sigma=0.,
        additive_noise_scale=0.,
        psf_sigma=0.,
    )
    peaks = IndexedPeaks(
        experiment.pattern_centre_px,
        offsets=np.asarray((complex(0, 0),)),
        hkls=np.asarray(((0, 0, 0),)),
        weights=np.asarray((1.,)),
    )
    frame = build_frame(
        experiment.frame_shape,
        experiment.pattern_centre_px,
        peaks.offsets,
        experiment.radius_px,
        minor=experiment.ellipse_minor,
        orientation=experiment.ellipse_orientation,
        intensities=peaks.weights,
        params=frame_params,
    )
    cy, cx = experiment.cyx
    signal = frame[int(cy), :].copy()
    signal = np.floor(signal)
    signal /= signal.max()
    fwhm, _, _, _ = peak_widths(signal, (int(cx),))
    assert_allclose(fwhm, radius * 2, rtol=0.01, atol=1.)


@pytest.mark.parametrize("frame_shape", (
    (512, 512),
    (511, 511),
))
@pytest.mark.parametrize("radius", (5, 12, 17))
@pytest.mark.parametrize("centre", (
    complex(388, 157),
    complex(256, 256),
))
@pytest.mark.parametrize("offset", (
    complex(0, 0),
    complex(10, -5),
    complex(-5.8, 15.7),
))
def test_offset_position(frame_shape, radius, centre, offset):
    experiment = ExperimentInformation(
        frame_shape=frame_shape,
        pattern_scale_factor=200.,  # pixels / Å-1
        radius_px=radius,
        centre_px=centre,
    )
    frame_params = FrameParameters(
        intensity_from_radius=True,
        textured=False,
        poisson_frame=False,
        disk_blur_sigma=0.5,
        inelastic_scatter_sigma=0.,
        additive_noise_scale=0.,
        psf_sigma=0.,
    )
    peaks = IndexedPeaks(
        experiment.pattern_centre_px,
        offsets=np.asarray((offset,)),
        hkls=np.asarray(((0, 0, 0),)),
        weights=np.asarray((1.,)),
    )
    frame = build_frame(
        experiment.frame_shape,
        experiment.pattern_centre_px,
        peaks.offsets,
        experiment.radius_px,
        minor=experiment.ellipse_minor,
        orientation=experiment.ellipse_orientation,
        intensities=peaks.weights,
        params=frame_params,
    )

    com = center_of_mass(frame)
    assert_allclose(
        com,
        to_array(experiment.pattern_centre_px + offset),
        rtol=0.,
        atol=0.1,  # allow 0.1 px error
    )
