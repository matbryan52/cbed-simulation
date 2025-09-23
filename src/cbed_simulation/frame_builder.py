from typing import NamedTuple, Sequence
import numpy as np
import numpy as xp
from skimage.draw import ellipse
from skimage.filters import gaussian
from perlin_numpy import generate_perlin_noise_2d


def fourier_shift(image_fft: np.ndarray, shift: np.ndarray, out=None):
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


def gen_noise(shape, scale=None):
    if scale is None:
        scale = 32 * (min(shape) // 512)
    true_shape = tuple(s + (scale - (s % scale)) for s in shape)
    noise = generate_perlin_noise_2d(true_shape, (scale, scale))
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


def aa_disk(frame, cy, cx, major, scale, minor=None, orientation=0.):
    frame_shape = frame.shape
    if minor is None:
        minor = major
    minor, major = sorted((minor, major))
    rr, cc = ellipse(
        cy, cx, minor, major, shape=frame_shape, rotation=np.deg2rad(orientation)
    )
    frame[rr, cc] = scale


def g1g2_pattern(frame_shape, g1, g2):
    ming = min(np.abs(g1), np.abs(g2))
    maxdim = np.linalg.norm(frame_shape)
    maxidx = np.ceil(maxdim / ming)

    px_shifts = np.mgrid[-maxidx: maxidx + 1, -maxidx: maxidx + 1].reshape(2, -1).T
    px_shifts = px_shifts[:, 0] * g1 + px_shifts[:, 1] * g2
    return px_shifts


class FrameParameters(NamedTuple):
    sim_intensities: bool = True
    disk_brightness: float = 1.
    mask_gauss_sigma: float = 2.
    falloff_power: float = 4.
    inelastic_sigma: float = 4.
    frame_brightness: float = 40.
    frame_noise_scale: float = 0.1


def build_frame(
    frame_shape: tuple[int, int],
    centre: complex | None,
    offsets: Sequence[complex],
    r: int,
    minor: float | None = None,
    orientation: float = 0.,
    params: FrameParameters = FrameParameters(),
    intensities: np.ndarray | None = None,
    progress: bool = False,
) -> np.ndarray:
    p = params
    h, w = frame_shape
    buffer = r = int(np.round(r))
    if centre is None:
        centre = complex(*(np.asarray(frame_shape) // 2))
    expanded_centre = centre + complex(buffer, buffer)

    frame = np.zeros(
        (h + 2 * buffer, w + 2 * buffer)
    ).astype(dtype=np.float32)
    base_frame = frame.copy()
    texture = gen_noise(frame_shape)
    texture = np.pad(
        texture,
        ((buffer, buffer), (buffer, buffer))
    )
    assert texture.shape == frame.shape

    # Make base frame
    cy, cx = np.asarray(base_frame.shape) // 2
    base_centre = complex(cx, cy)
    aa_disk(base_frame, cy, cx, r, p.disk_brightness, minor, orientation)
    base_frame = gaussian(base_frame, sigma=p.mask_gauss_sigma)
    base_frame = xp.fft.fft2(xp.asarray(base_frame.copy()))

    # Make the radius / extinction map
    eh, ew = base_frame.shape
    ccx, ccy = expanded_centre.real, expanded_centre.imag
    radius = np.linalg.norm(
        np.stack(
            np.meshgrid(
                np.arange(eh) - ccy,
                np.arange(ew) - ccx,
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

    if intensities is None or (not p.sim_intensities):
        multiplier = radius.copy()
        multiplier **= p.falloff_power
        intensities = np.ones(offsets.shape)
    else:
        multiplier = np.ones_like(frame)
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
        valid_intensities.append(intensity)

    frame = xp.fft.ifft2(
        fourier_shift(
            xp.asarray(base_frame), valid_shifts,
        )
    ).real
    frame *= xp.asarray(valid_intensities)[:, np.newaxis, np.newaxis]
    frame *= xp.asarray(texture)[xp.newaxis, ...]
    frame *= xp.asarray(multiplier)[xp.newaxis, ...]
    frame = xp.max(frame, axis=0)
    frame *= p.frame_brightness
    try:
        frame = frame.get()
    except AttributeError:
        pass

    gauss_frame = gaussian(frame, sigma=p.inelastic_sigma)
    noise = np.random.poisson(np.clip(gauss_frame.ravel(), 0.001, np.inf))
    frame += noise.reshape(frame.shape)
    frame += (
        np.random.poisson(
            (radius ** 2) * frame.max() * p.frame_noise_scale,
        )
        .astype(int)
        .reshape(frame.shape)
    )
    return np.round(frame[buffer: -buffer, buffer: -buffer]).astype(int)
