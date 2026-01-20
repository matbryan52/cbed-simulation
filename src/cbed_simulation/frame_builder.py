from typing import NamedTuple, Sequence, Literal

import numpy as np
from skimage.draw import ellipse

from .utils import get_backend, to_numpy


# interpolant() and generate_perlin_noise_2d() copied from
# https://github.com/pvigier/perlin-numpy/blob/master/perlin_numpy/perlin2d.py
# (MIT licensed) and adapted to make it compatible with Cupy

def interpolant(t):
    return t*t*t*(t*(t*6 - 15) + 10)


def generate_perlin_noise_2d(
        shape, res, tileable=(False, False), interpolant=interpolant,
        xp=np
):
    """Generate a 2D numpy array of perlin noise.

    Args:
        shape: The shape of the generated array (tuple of two ints).
            This must be a multple of res.
        res: The number of periods of noise to generate along each
            axis (tuple of two ints). Note shape must be a multiple of
            res.
        tileable: If the noise should be tileable along each axis
            (tuple of two bools). Defaults to (False, False).
        interpolant: The interpolation function, defaults to
            t*t*t*(t*(t*6 - 15) + 10).

    Returns:
        A numpy array of shape shape with the generated noise.

    Raises:
        ValueError: If shape is not a multiple of res.
    """
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = xp.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]]\
             .transpose(1, 2, 0) % 1
    # Gradients
    angles = 2*xp.pi*xp.random.rand(res[0]+1, res[1]+1)
    gradients = xp.dstack((xp.cos(angles), xp.sin(angles)))
    if tileable[0]:
        gradients[-1, :] = gradients[0, :]
    if tileable[1]:
        gradients[:, -1] = gradients[:, 0]

    gradients = gradients.repeat(d[0], axis=0).repeat(d[1], axis=1)

    g00 = gradients[:-d[0], :-d[1]]
    g10 = gradients[d[0]:, :-d[1]]
    g01 = gradients[:-d[0], d[1]:]
    g11 = gradients[d[0]:, d[1]:]

    # Ramps
    n00 = xp.sum(
        xp.dstack((grid[:, :, 0], grid[:, :, 1])) * g00,
        axis=2,
    )
    n10 = xp.sum(
        xp.dstack((grid[:, :, 0] - 1, grid[:, :, 1])) * g10,
        axis=2,
    )
    n01 = xp.sum(
        xp.dstack((grid[:, :, 0], grid[:, :, 1] - 1)) * g01,
        axis=2,
    )
    n11 = xp.sum(
        xp.dstack((grid[:, :, 0] - 1, grid[:, :, 1] - 1)) * g11,
        axis=2,
    )

    # Interpolation
    t = interpolant(grid)
    n0 = n00 * (1 - t[:, :, 0]) + t[:, :, 0] * n10
    n1 = n01 * (1 - t[:, :, 0]) + t[:, :, 0] * n11
    return xp.sqrt(2) * ((1 - t[:, :, 1]) * n0 + t[:, :, 1] * n1)


# Until marker ***
# Taken from ptychography40
# https://github.com/Ptychography-4-0/ptychography/blob/master/src/ptychography40/reconstruction/common.py
# 3cecf896f409a6f53ad782b7d57cab7438e995b9
# Relicensed from original author Dieter Weber under MIT license for this project

def offset(s1, s2):
    o1, ss1 = s1
    o2, ss2 = s2
    return o2 - o1


def shift_by(sl, shift):
    origin, shape = sl
    return (
        origin + shift,
        shape
    )


def shift_to(s1, origin):
    o1, ss1 = s1
    return (
        origin,
        ss1
    )


def intersection(s1, s2):
    o1, ss1 = s1
    o2, ss2 = s2
    # Adapted from libertem.common.slice
    new_origin = np.maximum(o1, o2)
    new_shape = np.minimum(
        (o1 + ss1) - new_origin,
        (o2 + ss2) - new_origin,
    )
    new_shape = np.maximum(0, new_shape)
    return (new_origin, new_shape)


def get_shifted(arr_shape, tile_origin, tile_shape, shift):
    '''
    Calculate the slices to cut out a shifted part of a 2D source
    array and place it into a target array, including tiling support.

    This works with negative and positive integer shifts.
    '''
    # TODO this could be adapted for full sig, nav, n-D etc support
    # and included as a method in Slice?
    full_slice = (np.array((0, 0)), arr_shape)
    tileslice = (tile_origin, tile_shape)
    shifted = shift_by(tileslice, shift)
    isect = intersection(full_slice, shifted)
    if np.prod(isect[1]) == 0:
        return (
            np.array([(0, 0), (0, 0)]),
            np.array([0, 0])
        )
    # We measure by how much we have clipped the zero point
    # This is zero if we didn't shift into the negative region beyond the original array
    clip = offset(shifted, isect)
    # Now we move the intersection to (0, 0) plus the amount we clipped
    # so that the overlap region is moved by the correct amount, in total
    targetslice = shift_by(shift_to(isect, np.array((0, 0))), clip)
    start = targetslice[0]
    length = targetslice[1]
    target_tup = np.stack((start, start+length), axis=1)
    offsets = isect[0] - targetslice[0]
    return (target_tup, offsets)


def to_slices(target_tup, offsets):
    target_slice = tuple(slice(s[0], s[1]) for s in target_tup)
    source_slice = tuple(slice(s[0] + o, s[1] + o) for (s, o) in zip(target_tup, offsets))
    return (target_slice, source_slice)


def fourier_shift(image_fft: np.ndarray, shift: np.ndarray, out=None, xp=np):
    """
    Implements Fourier shifting like scipy.ndimage
    but with numpy broadcasting to apply multiple
    shifts to the same image and get a stacked result
    """
    assert image_fft.ndim == 2
    h, w = input_shape = image_fft.shape
    shift = xp.asarray(shift).astype(xp.float32)
    assert shift.shape[-1] == 2
    expanded = False
    if shift.ndim == 1:
        shift = shift[xp.newaxis, ...]
        expanded = True
    assert shift.ndim == 2
    shift_precalc = (
        -2 * xp.pi * shift
        / xp.asarray((input_shape,))
    )
    h_idx = xp.fft.fftfreq(h, d=1 / h)
    v_idx = xp.fft.fftfreq(w, d=1 / w)
    p_idx = xp.stack(
        xp.meshgrid(h_idx, v_idx, indexing="ij"),
        axis=-1
    )
    p_shift = p_idx[xp.newaxis, ...] * shift_precalc[:, xp.newaxis, xp.newaxis, :]
    phase_shifts = p_shift.sum(axis=-1)
    if out is None:
        out = xp.empty(phase_shifts.shape, dtype=image_fft.dtype)
    xp.multiply(
        image_fft[xp.newaxis, ...],
        xp.exp(
            1j * phase_shifts
        ),
        out=out,
    )
    if expanded:
        out = out[0]
    return out


def gen_noise(shape, scale=None, xp=np):
    if scale is None:
        scale = min(shape)
    true_shape = tuple(s + (scale - (s % scale)) for s in shape)
    noise = generate_perlin_noise_2d(true_shape, (scale, scale), xp=xp)
    noise -= noise.min()
    noise /= noise.max()
    noise += 1
    return noise[:shape[0], :shape[1]]


def _rotation(angle):
    return np.asarray([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle),  np.cos(angle)],
    ])


def apply_strain(e_xx, e_xy, e_yy, e_rot, g1, g2):
    R = _rotation(e_rot)
    epsilon = np.asarray([
        [e_xx, e_xy],
        [e_xy, e_yy],
    ])
    G0 = np.asarray([
        [g1.real,  g2.real],
        [g1.imag,  g2.imag],
    ])
    Gstrained_p = np.linalg.inv(
            R @ (
                epsilon + np.eye(2)
            ) @ np.linalg.inv(G0.T)
    ).T
    return complex(*Gstrained_p[:, 0]), complex(*Gstrained_p[:, 1])


def draw_ellipse(frame_shape, cy, cx, major, scale, minor=None, orientation=0.):
    h, w = frame_shape
    frame = np.zeros((int(h), int(w)), dtype=np.float32)
    if minor is None:
        minor = major
    minor, major = sorted((minor, major))
    rr, cc = ellipse(
        cy, cx, minor, major,
        shape=frame.shape, rotation=np.deg2rad(orientation)
    )
    frame[rr, cc] = scale
    return frame


def g1g2_pattern(frame_shape, g1, g2):
    ming = min(np.abs(g1), np.abs(g2))
    maxdim = np.linalg.norm(frame_shape)
    maxidx = np.ceil(maxdim / ming)

    px_shifts = np.mgrid[-maxidx: maxidx + 1, -maxidx: maxidx + 1].reshape(2, -1).T
    px_shifts = px_shifts[:, 0] * g1 + px_shifts[:, 1] * g2
    return px_shifts


class FrameParameters(NamedTuple):
    disk_brightness: float = 1.
    # the default blur value effectively anti-aliases
    # the disk without changing the radius more than 1 px
    disk_blur_sigma: float = 0.5
    intensity_from_radius: bool = False
    intensity_falloff_power: float = 4.
    textured: bool = True
    frame_brightness: float = 40.
    inelastic_scatter_sigma: float = 4.
    additive_noise_scale: float = 0.1
    multiplicative_noise_scale: float = 0.


def build_frame(
    frame_shape: tuple[int, int],
    centre: complex | None,
    offsets: Sequence[complex],
    r: int,
    minor: float | None = None,
    orientation: float = 0.,
    params: FrameParameters = FrameParameters(),
    intensities: np.ndarray | None = None,
    backend: Literal["cupy", "cpu"] = "cpu",
) -> np.ndarray:
    xp, ndimage = get_backend(backend)
    xp: np

    p = params
    h, w = frame_shape
    buffer = r = int(np.round(r))
    if centre is None:
        centre = complex(*(np.asarray(frame_shape) // 2))
    assert centre.real == int(centre.real), "centre position must be integer"
    assert centre.imag == int(centre.imag), "centre position must be integer"
    expanded_centre = centre + complex(buffer, buffer)

    frame = np.zeros(
        (h + 2 * buffer, w + 2 * buffer)
    ).astype(dtype=np.float32)
    if p.textured:
        texture = gen_noise(frame_shape, xp=xp)
        texture = xp.pad(
            texture,
            ((buffer, buffer), (buffer, buffer))
        )
        assert texture.shape == frame.shape

    if minor is not None:
        bounding_r = max(r, minor)
    else:
        bounding_r = r
    cropped_size = 2 * bounding_r + 2
    if p.disk_blur_sigma > 0:
        cropped_size += 2 * 3 * p.disk_blur_sigma
    cropped_size = int(xp.ceil(cropped_size))
    # Make base frame
    cy = cx = cropped_size // 2
    base_centre = complex(cx, cy)
    base_frame = xp.array(draw_ellipse(
        (cropped_size, cropped_size), cy, cx, r, p.disk_brightness, minor, orientation
    ))

    if p.disk_blur_sigma > 0:
        base_frame = ndimage.gaussian_filter(base_frame, sigma=p.disk_blur_sigma)
    base_frame = xp.fft.fft2(xp.asarray(base_frame.copy()))

    # Make the radius / extinction map
    eh, ew = frame.shape
    ccx, ccy = expanded_centre.real, expanded_centre.imag
    radius = xp.linalg.norm(
        xp.stack(
            xp.meshgrid(
                xp.arange(eh) - ccy,
                xp.arange(ew) - ccx,
                indexing="ij",
            ),
            axis=0
        ),
        axis=0,
    )
    max_r = min(h, w) / 2.
    radius /= max_r
    radius -= radius.max()
    radius *= -1

    if intensities is None or p.intensity_from_radius:
        radius_adjusted = radius.copy()
        radius_adjusted **= p.intensity_falloff_power
    else:
        intensities = intensities.copy()
        intensities /= intensities.max()

    valid_shifts = []
    valid_intensities = []
    for shift, intensity in zip(offsets, intensities):
        real_pos = (expanded_centre + shift)
        if not (0 <= real_pos.real < ew):
            continue
        if not (0 <= real_pos.imag <= eh):
            continue

        this_shift = real_pos - base_centre
        valid_shifts.append((this_shift.imag, this_shift.real))
        if intensities is None or p.intensity_from_radius:
            intensity = radius_adjusted[int(round(real_pos.imag)), int(round(real_pos.real))]
        valid_intensities.append(intensity)

    valid_shifts = xp.asarray(valid_shifts)
    pixel_shifts, subpixel_shifts = xp.divmod(valid_shifts, 1)

    subpixel_frame = xp.fft.ifft2(
        fourier_shift(
            xp.asarray(base_frame), subpixel_shifts,
            xp=xp,
        )
    ).real

    subpixel_frame *= xp.asarray(valid_intensities)[:, np.newaxis, np.newaxis]

    frame = xp.zeros(shape=subpixel_frame.shape[:1] + frame.shape, dtype=subpixel_frame.dtype)

    # Do slice computations on CPU
    pixel_shifts = to_numpy(pixel_shifts)
    for i, pixel_shift in enumerate(pixel_shifts):
        target_tup, offsets = get_shifted(
            arr_shape=np.array(subpixel_frame.shape[1:]),
            tile_origin=np.array((0, 0)),
            tile_shape=np.array(frame.shape[1:]),
            # No idea why negative but it is what it is...
            shift=(-pixel_shift).astype(int),
        )
        target, source = to_slices(target_tup, offsets)
        frame[i][target] = subpixel_frame[i][source]

    if p.textured:
        frame *= xp.asarray(texture)[xp.newaxis, ...]
    frame = xp.max(frame, axis=0)
    frame *= p.frame_brightness

    if p.inelastic_scatter_sigma > 0.:
        gauss_frame = ndimage.gaussian_filter(frame, sigma=p.inelastic_scatter_sigma)
        noise = xp.random.poisson(xp.clip(gauss_frame.ravel(), 0.001, xp.inf))
        frame += noise.reshape(frame.shape)
    if p.additive_noise_scale > 0.:
        frame += (
            xp.random.poisson(
                (radius ** 2) * frame.max() * p.additive_noise_scale,
            )
            .astype(int)
            .reshape(frame.shape)
        )
    if p.multiplicative_noise_scale:
        frame = xp.random.poisson(frame * p.multiplicative_noise_scale)
    return xp.round(frame[buffer: -buffer, buffer: -buffer]).astype(int)
