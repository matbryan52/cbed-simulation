import os
import types
import copy
import pathlib
from typing import Literal, TYPE_CHECKING
import numpy as np
from numpy.typing import ArrayLike
import scipy.ndimage as ndimage
from scipy import constants
from diffpy.structure.parsers.p_cif import P_cif
from orix.crystal_map import Phase
from orix.vector import Vector3d, Miller
from orix.quaternion import Rotation
from orix.io.plugins.ang import file_reader

cp = None
ndimage_cp = None
try:
    import cupy as cp
    import cupyx.scipy.ndimage as ndimage_cp
except ModuleNotFoundError:
    pass


if TYPE_CHECKING:
    from .crystal_orientation import IndexedPeaks


def to_complex(array, xp=np) -> complex | np.ndarray[complex]:
    """
    Convert [..., (y, x)] to x + y * 1j complex
    """
    if xp.asarray(array).size == 2:
        return complex(*(array[::-1]))
    return xp.array(array[..., :, 1] + array[..., 0] * 1j)


def to_array(complex_array, xp=np):
    """
    Convert x + y * 1j to array [..., (y, x)]
    """
    if xp.asarray(complex_array).size == 1 and xp.iscomplex(complex_array).all():
        return xp.asarray((complex_array.imag, complex_array.real))
    complex_array = xp.asarray(complex_array)
    return xp.stack(
        (
            complex_array.imag,
            complex_array.real,
        ),
        axis=-1,
    )


def scale_and_rotate(offsets, scale: float, rotation: float):
    return offsets * scale * np.exp(1j * np.deg2rad(rotation))


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


def get_ndimage(backend: types.ModuleType):
    if backend is np:
        return ndimage
    elif backend is cp:
        if ndimage_cp is None:
            raise ModuleNotFoundError("Missing functioning cupyx for backend")
        return ndimage_cp
    else:
        raise ValueError("Unrecognized backend for ndimage")


def get_backend(backend: Literal["cupy", "cpu"] | types.ModuleType):
    if isinstance(backend, types.ModuleType):
        pass
    elif backend == "cpu":
        backend = np
    elif backend == "cupy":
        if cp is None:
            raise ModuleNotFoundError("Missing functioning cupy for backend")
        backend = cp
    else:
        raise ValueError(f"Unrecognized backend {backend}")
    return backend, get_ndimage(backend)


def to_numpy(arr: ArrayLike):
    if arr.device != 'cpu':
        return arr.get()
    else:
        return arr


def overlay_peaks(
    peaks1: "IndexedPeaks",
    peaks2: "IndexedPeaks",
    names=(1, 2),
    highlight=((0, 0, 0),),
    savepath=None,
):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 8))
    for peaks, color, baseline, name in zip(
        (peaks1, peaks2),
        ("k", "r"),
        ("top", "bottom"),
        names,
    ):
        if peaks is None:
            continue
        ax.plot(peaks.offsets.real, peaks.offsets.imag, color + 'x', label=f"{name}")
        for idx, (offset, hkl) in enumerate(zip(peaks.offsets, peaks.hkls)):
            ax.text(
                offset.real,
                offset.imag,
                f"{idx}{hkl}",
                color=color,
                verticalalignment=baseline,
                horizontalalignment="center"
            )
    for hkl in highlight:
        pos = peaks1.spot_position(hkl, centre_zero=True)
        ax.plot(pos.real, pos.imag, "go")
    if savepath is not None:
        plt.savefig(savepath)
    return fig, ax


def _rotation_z_to_vecs(vecs: Vector3d) -> Rotation:
    """
    Return Rotation objects mapping z → v for each v in vecs,
    with a consistent φ₁ = 0 reference in Bunge (ZXZ) convention.

    Uses the same decomposition as orix's get_sample_reduced_fundamental:
      φ₁ = 0                              — fixed in-plane reference
      Φ  = arccos(vz)                     — polar angle from z
      φ₂ = (π/2 − arctan2(vy, vx)) % 2π  — azimuthal angle

    The constraint φ₁ = 0 keeps the in-plane reference constant across
    all directions, so the γ recovered during correlation is directly
    comparable between any two plan entries and equals the Bunge φ₁ of
    the full orientation.
    """
    v = vecs.unit.data                          # (..., 3), normalised
    vz = np.clip(v[..., 2], -1.0, 1.0)
    Phi = np.arccos(vz)                         # polar angle
    azimuth = np.arctan2(v[..., 1], v[..., 0])
    phi1 = np.zeros_like(Phi)
    phi2 = (np.pi / 2 - azimuth) % (2 * np.pi)
    return Rotation.from_euler(np.stack([phi1, Phi, phi2], axis=-1))


def orientation_for_hkl(phase: Phase, hkl: tuple[int, int, int]):
    """
    Get the crystal2lab Rotation from Z-axis to an hkl
    """
    miller = Miller(hkl=hkl, phase=phase)
    vec = Vector3d((miller.x.item(), miller.y.item(), miller.z.item()))
    # NOTE this could be replaced with Miller(hkl=(0, 0, 1)) to be more correct
    # but this would need an equivalent change in strain_orientation_mapping
    return _rotation_z_to_vecs(vec)


def cif_to_phase(cif_path: os.PathLike):
    cif_path = pathlib.Path(cif_path)
    with cif_path.open('r') as fp:
        cif_str = fp.read()
    cif = P_cif()
    structure = cif.parse(cif_str)
    spacegroup = cif.spacegroup
    return Phase(
        cif_path.stem,
        space_group=spacegroup,
        structure=copy.deepcopy(structure),
    )


def electron_wavelength_angstrom(E_eV: float) -> float:
    m = constants.m_e
    e = constants.elementary_charge
    c = constants.c
    h = constants.h
    return h / np.sqrt(2 * m * e * E_eV) / np.sqrt(1 + e * E_eV / 2 / m / c**2) * 10**10
