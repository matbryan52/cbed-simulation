import os
import numpy as np
from orix.vector import Vector3d
from orix.quaternion import Rotation
from orix.io.plugins.ang import file_reader


def to_complex(array) -> complex | np.ndarray[complex]:
    """
    Convert [..., (y, x)] to x + y * 1j complex
    """
    if np.asarray(array).size == 2:
        return complex(*(array[::-1]))
    return array[..., :, 1] + array[..., 0] * 1j


def to_array(complex_array):
    """
    Convert x + y * 1j to array [..., (y, x)]
    """
    if np.asarray(complex_array).size == 1 and np.iscomplex(complex_array).all():
        return np.asarray((complex_array.imag, complex_array.real))
    complex_array = np.asarray(complex_array)
    return np.stack(
        (
            complex_array.imag,
            complex_array.real,
        ),
        axis=-1,
    )


def to_miller_ltx(h: int, k: int, l: int):  # noqa
    _to_miller = lambda v: f"{v}" if v >= 0 else "\\bar{" + f"{abs(v)}" + "}"  # noqa
    return f"${_to_miller(h)}{_to_miller(k)}{_to_miller(l)}$"


def flip_y(rot: Rotation):
    return Rotation.from_axes_angles(
        rot * Vector3d.yvector(), 180., degrees=True,
    ) * rot


def load_ang(filepath: os.PathLike, do_flip_y: bool = True):
    xtal_map = file_reader(filepath)
    if do_flip_y:
        xtal_map._rotations = flip_y(xtal_map.rotations)
    return xtal_map
