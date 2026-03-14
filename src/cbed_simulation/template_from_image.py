import numpy as np
from typing import Literal
from skimage.filters import sobel
from skimage.transform import rescale
from scipy.ndimage import center_of_mass, fourier_shift


def r_map(im_shape, cyx):
    h, w = im_shape
    cy, cx = cyx
    xx, yy = np.meshgrid(
        np.arange(h) - cy, np.arange(w) - cx,
        indexing="ij",
    )
    return np.sqrt(xx ** 2 + yy ** 2)


def sigmoid_2d(im_shape, cyx, r):
    radii = r_map(im_shape, cyx)
    A = 1
    K = 0
    B = 1
    return A + ((K - A) / (1 + np.exp(-1 * B * (radii - r))) ** 2)


def fourier_shift_img(image, shift):
    return np.fft.ifft2(
        fourier_shift(
            np.fft.fft2(image),
            shift,
        )
    ).real


def _crop_or_insert(dest_dd, source_dd):
    left = np.s_[:]
    right = np.s_[:dest_dd]
    if source_dd < dest_dd:
        left = np.s_[:source_dd]
        right = np.s_[:]
    return left, right


def crop_or_insert(dest: tuple[int, int], source: tuple[int, int]):
    ihh, iww = dest
    shh, sww = source
    lh, rh = _crop_or_insert(ihh, shh)
    lw, rw = _crop_or_insert(iww, sww)
    return (lh, lw), (rh, rw)


def template_from_vacuum(
    frame: np.ndarray,
    cyx: tuple[float, float],
    r_estimate: float,
    beam_rescale_factor: float = 0.95,
    edge_rescale_factor: float | None = 1.025,
    edge_strength: float = 1.,
    clip_max_frac: float = 0.5,
    sigmoid_taper_frac: float = 1.5,
):
    # equivalent of get_vacuum_probe / add_vacuum_region
    vac_frame = frame * sigmoid_2d(
        frame.shape, cyx, r_estimate * sigmoid_taper_frac
    )
    # flatten disk centre and norm to 1
    max_val = np.max(vac_frame)
    vac_frame = np.clip(
        vac_frame, 0., clip_max_frac * max_val,
    )
    vac_frame /= vac_frame.max()
    orig_com = center_of_mass(vac_frame)

    if beam_rescale_factor != 1.:
        rescaled_frame = rescale(vac_frame, beam_rescale_factor)
        rescaled_com = center_of_mass(rescaled_frame)
        shifted_rescaled = fourier_shift_img(
            rescaled_frame, np.asarray(orig_com) - np.asarray(rescaled_com),
        )
        left, right = crop_or_insert(vac_frame.shape, shifted_rescaled.shape)
        vac_frame[left] = shifted_rescaled[right]

    template = vac_frame.copy()

    if edge_rescale_factor is not None:
        orig_com = center_of_mass(vac_frame)
        rescaled_frame = rescale(vac_frame, edge_rescale_factor)
        rescaled_com = center_of_mass(rescaled_frame)
        edge_map = sobel(rescaled_frame)
        shifted_edge_map = fourier_shift_img(
            edge_map, np.asarray(orig_com) - np.asarray(rescaled_com),
        )
        left, right = crop_or_insert(template.shape, shifted_edge_map.shape)
        template[left] -= (edge_strength * shifted_edge_map[right])
    return template


def shift_probe(template, cyx: tuple[float, float], shifted: Literal["fourier", "bilinear"]):
    from py4DSTEM.process.utils import get_shifted_ar
    assert shifted.lower() in ("fourier", "bilinear")
    cy, cx = cyx
    return get_shifted_ar(
        template, -cy, -cx, bilinear=shifted.lower() == "bilinear",
    )


def com_crop(frame: np.ndarray, size: int):
    cy, cx = np.round(
        center_of_mass(frame - frame.min())
    ).astype(int)
    hsize = size // 2
    return np.s_[
        cy - hsize: cy + hsize,
        cx - hsize: cx + hsize,
    ]


def subpixel_com_crop(template: np.ndarray, size: int):
    crop = com_crop(template, size)
    crop = template[crop]
    h2, w2 = (crop.shape[0] - 1) / 2., (crop.shape[1] - 1) / 2.
    cy, cx = center_of_mass(crop - crop.min())
    return fourier_shift_img(
        crop, (h2 - cy, w2 - cx),
    )
