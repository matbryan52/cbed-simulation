import os
import pathlib
import types
from typing import NamedTuple, Literal
import numpy as np

from orix.quaternion import Rotation, Quaternion
from orix.crystal_map import Phase
from diffsims.generators.simulation_generator import SimulationGenerator

from .utils import (
    to_complex,
    to_array,
    get_backend,
    to_numpy,
    orientation_for_hkl,
    cif_to_phase,
    scale_and_rotate,
    electron_wavelength_angstrom,
)
from .distortions import DistortionConfig, apply_distortion
from .frame_builder import build_frame, FrameParameters


class LatticeMultipliers(NamedTuple):
    a: float = 1.
    b: float = 1.
    c: float = 1.
    alpha: float = 1.
    beta: float = 1.
    gamma: float = 1.

    def apply_ase(self, atoms):
        new_atoms = atoms.copy()
        new_atoms.cell = atoms.cell.copy()
        cellpar = new_atoms.cell.cellpar().copy()
        cellpar *= np.asarray(self)
        new_atoms.set_cell(cellpar, scale_atoms=True)
        return new_atoms

    def apply_diffsims(self, phase: Phase):
        new_phase = phase.deepcopy()
        cellpar = np.asarray(new_phase.structure.lattice.abcABG())
        cellpar *= np.asarray(self)
        new_phase.structure.lattice.setLatPar(*cellpar)
        return new_phase


class EulerAngles(NamedTuple):
    phi1: float
    Phi: float
    phi2: float


class GVecs(NamedTuple):
    g1: complex
    g2: complex

    def to_array(self):
        return np.asarray([
            [self.g1.imag, self.g1.real],
            [self.g2.imag, self.g2.real],
        ])


class ExperimentInformation(NamedTuple):
    frame_shape: tuple[int, int]
    pattern_scale_factor: float  # px / Å-1
    radius_px: int  # spot major axis
    centre_px: complex | None = None  # (x + i * y), frame centre by default
    ellipse_minor: float | None = None  # radius_px is major axis
    ellipse_orientation: float = 0.  # degrees
    voltage_kv: float = 200.
    precession_angle: float = 1.
    debye_waller_factors: dict | None = None

    @classmethod
    def default(cls):
        return cls(
            frame_shape=(512, 512),
            pattern_scale_factor=120,
            radius_px=12,
        )

    @classmethod
    def from_tem_params(
        cls,
        camera_length_m: float,
        semiconv_mrad: float,
        voltage_kv: float,
        frame_shape: tuple[int, int],
        pixelsize_um: float,
        **kwargs,
    ):
        """
        Create ExperimentInformation from physical parameters of acquisition
        """
        semiconv = semiconv_mrad * 1e-3
        pixelsize = pixelsize_um * 1e-6
        voltage = voltage_kv * 1e3
        lam_a = electron_wavelength_angstrom(voltage)
        pattern_scale_factor = camera_length_m * lam_a / pixelsize  # px / Å-1
        radius_px = np.round(semiconv * pattern_scale_factor / lam_a).astype(int)
        return cls(
            frame_shape=frame_shape,
            pattern_scale_factor=pattern_scale_factor.item(),
            radius_px=radius_px.item(),
            voltage_kv=voltage_kv,
            **kwargs,
        )

    def _px_to_angle_mrad(self, radius: float):
        lam_a = electron_wavelength_angstrom(self.voltage_kv * 1e3)
        return (radius * lam_a / self.pattern_scale_factor) * 1e3

    @property
    def semiconv_mrad(self):
        return self._px_to_angle_mrad(self.radius_px)

    def _max_distance_px(self):
        br = complex(*self.frame_shape[::-1])
        tr = br.real + 0j
        bl = br.imag * 1j
        return max(
            abs(self.pattern_centre_px),
            abs(br - self.pattern_centre_px),
            abs(tr - self.pattern_centre_px),
            abs(bl - self.pattern_centre_px),
        )

    @property
    def max_extent(self) -> float:
        """
        Maximum diffraction vector in Å-1
        """
        return self._max_distance_px() * self.pixelsize

    @property
    def max_angle(self) -> float:
        """
        Maximum diffraction angle in mrad
        """
        return self._px_to_angle_mrad(self._max_distance_px())

    @property
    def pixelsize(self) -> float:
        """
        Pixelsize in Å-1 / px
        """
        return 1 / self.pattern_scale_factor

    @property
    def spotsize(self) -> float:
        """
        Spot size in Å-1
        """
        return self.radius_px * self.pixelsize

    def modify(self, **kwargs):
        params = self._asdict()
        params.update(kwargs)
        return type(self)(**params)

    @property
    def frame_centre_px(self):
        hh, ww = self.frame_shape  # (yy, xx)
        return complex(ww / 2., hh / 2.)  # x-real, y-imag

    @property
    def pattern_centre_px(self):
        if self.centre_px is None:
            return self.frame_centre_px
        return self.centre_px

    @property
    def cyx(self):
        return self.pattern_centre_px.imag, self.pattern_centre_px.real


class IndexedPeaks(NamedTuple):
    pos_000: complex
    offsets: np.ndarray[complex]
    hkls: np.ndarray
    weights: np.ndarray

    @property
    def size(self):
        return self.offsets.size

    @property
    def peaks(self):
        return self.pos_000 + self.offsets

    def spot_index(self, hkl: tuple[int, int, int]) -> int:
        hkl = tuple(hkl)
        for idx, _hkl in enumerate(self.hkls):
            if tuple(_hkl) == hkl:
                return idx
        raise ValueError(f"hkl {hkl} not found")

    def spot_position(self, hkl: tuple[int, int, int], centre_zero: bool = False) -> complex:
        idx = self.spot_index(hkl)
        if centre_zero:
            return self.offsets[idx]
        return self.peaks[idx]

    def to_array(self, centre_zero: bool = False):
        if centre_zero:
            return to_array(self.offsets)
        return to_array(self.peaks)

    def modify_intensities(self, power: float):
        self.weights[:] **= power

    def modify_000_intensity(self, multiply: float):
        index_000 = self.spot_index((0, 0, 0))
        self.weights[index_000] *= multiply

    def apply_mask(self, mask: np.ndarray):
        return type(self)(
            self.pos_000,
            self.offsets[mask],
            self.hkls[mask],
            self.weights[mask],
        )

    def match_peaks(self, other: 'IndexedPeaks'):
        """
        Return self and other, containing only common hkl peaks
        """
        this_hlks = tuple(tuple(hkl) for hkl in self.hkls)
        other_hlks = tuple(tuple(hkl) for hkl in other.hkls)
        keep = {}
        for i, hkl in enumerate(this_hlks):
            try:
                other_i = other_hlks.index(hkl)
            except ValueError:
                continue
            keep[i] = other_i
        keep_this = np.asarray(tuple(keep.keys())).astype(int)
        keep_other = np.asarray(tuple(keep.values())).astype(int)
        new_this = type(self)(
            self.pos_000,
            self.offsets[keep_this],
            self.hkls[keep_this],
            self.weights[keep_this],
        )
        new_other = type(other)(
            other.pos_000,
            other.offsets[keep_other],
            other.hkls[keep_other],
            other.weights[keep_other],
        )
        return new_this, new_other

    def plot(
        self,
        fig,
        ax,
        interactive: bool = True,
        point_alpha: float = 1.,
    ):
        from .interactive_miller_plot import plot_pattern
        plot_pattern(
            fig,
            ax,
            frame=None,
            max_extent=None,
            spots=to_numpy(to_array(self.peaks)[:, ::-1]),
            intensity=2,
            millers=self.hkls,
            scatter_alpha=point_alpha,
            interactive=interactive,
        )

    def to_numpy(self):
        return type(self)(
            complex(self.pos_000),
            to_numpy(self.offsets),
            to_numpy(self.hkls),
            to_numpy(self.weights),
        )


class SimulatedPeaks(IndexedPeaks):
    def to_pixels(
        self,
        experiment: ExperimentInformation,
        clip: bool = False,
    ) -> IndexedPeaks:
        offsets = scale_and_rotate(
            self.offsets,
            experiment.pattern_scale_factor,
            0.,
        )
        contained = np.s_[:]
        if clip:
            positions_px = experiment.pattern_centre_px + offsets
            contained = (
                (positions_px.real >= 0)
                & (positions_px.real < experiment.frame_shape[1] - 1)
                & (positions_px.imag >= 0)
                & (positions_px.imag < experiment.frame_shape[0] - 1)
            )
        return IndexedPeaks(
            experiment.pattern_centre_px,
            offsets[contained],
            self.hkls[contained],
            self.weights[contained],
        )

    def angle_of(self, hkl: tuple[int, int, int], rad: bool = True) -> float:
        return np.angle(self.spot_position(hkl, centre_zero=True), deg=not rad)


class EllipseDef(NamedTuple):
    a: float
    b: float
    theta: float


class InversePixelsize(NamedTuple):
    qx: float
    qy: float


class Pixelsize(NamedTuple):
    xscale: float
    yscale: float

    def inverse(self):
        return InversePixelsize(1 / self.xscale, 1 / self.yscale)


class OrientedPhase(NamedTuple):
    cif_path: os.PathLike
    phase: Phase
    orientation: Rotation

    @classmethod
    def from_cif(
        cls,
        cif_path: os.PathLike,
        orientation: tuple[float, float, float] | Rotation | None = None,
        zone_axis: tuple[int, int, int] | None = None,
        in_plane_rot: float = 0.,  # degrees
    ):
        cif_path = pathlib.Path(cif_path).absolute()
        phase = cif_to_phase(cif_path)
        if orientation is None and zone_axis is None:
            orientation = (0., 0., 0.)  # null Euler angles
        orientation = cls._get_orientation(
            phase, orientation, zone_axis, in_plane_rot
        )
        return cls(
            cif_path,
            phase,
            orientation,
        )

    @staticmethod
    def _get_orientation(
        phase: Phase,
        orientation: tuple[float, float, float] | Rotation | None = None,
        zone_axis: tuple[int, int, int] | None = None,
        in_plane_rot: float = 0.,
    ):
        assert not (
            (orientation is not None)
            and (zone_axis is not None)
        ), "Can only supply one of orientation or zone_axis"
        if zone_axis is not None:
            orientation = orientation_for_hkl(
                phase, zone_axis,
            )
        elif isinstance(orientation, (list, tuple, np.ndarray)):
            orientation = Rotation.from_euler(
                orientation,
                direction="crystal2lab",
                degrees=True,
            )
        elif isinstance(orientation, Quaternion):
            pass
        else:
            raise TypeError("Unrecognized orientation type")
        assert isinstance(orientation, Quaternion)
        if in_plane_rot != 0.:
            orientation = Rotation.from_euler(
                (0., 0., in_plane_rot),
                # c2l results in a negative rotation when comparing
                # the same peak from a rotated phase
                # direction="crystal2lab",
                degrees=True,
            ) * orientation
        return orientation

    @property
    def atoms(self):
        from ase.io import read as read_atoms
        return read_atoms(self.cif_path)

    def with_rot(
        self,
        orientation: tuple[float, float, float] | Rotation | None = None,
        zone_axis: tuple[int, int, int] | None = None,
        in_plane_rot: float = 0.,
    ):
        assert (
            (orientation is not None) or (zone_axis is not None)
        ), "Must supply one of orientation or zone_axis"
        return type(self)(
            self.cif_path,
            self.phase,
            self._get_orientation(
                self.phase,
                orientation,
                zone_axis,
                in_plane_rot,
            ),
        )

    def _kinematical_sim(
        self,
        experiment: ExperimentInformation,
        max_excitation_error: float | None = None,
        max_extent: float | None = None,
        lattice_mod: LatticeMultipliers = LatticeMultipliers(),
        backend: Literal["cupy", "cpu"] | types.ModuleType = "cpu",
    ):
        xp, _ = get_backend(backend)

        if max_excitation_error is None:
            max_excitation_error = 0.03
        gen = SimulationGenerator(
            accelerating_voltage=experiment.voltage_kv,
            precession_angle=experiment.precession_angle,
        )
        sim = gen.calculate_diffraction2d(
            phase=lattice_mod.apply_diffsims(self.phase),
            rotation=self.orientation.inv(),
            reciprocal_radius=max_extent,
            max_excitation_error=max_excitation_error,
            with_direct_beam=True,
            debye_waller_factors=experiment.debye_waller_factors,
        )
        params = dict(
            in_plane_angle=0.,  # anti-clockwise rotation of simulated peaks in degrees
            direct_beam_position=(0., 0.),
            mirrored=False,  # mirror the pattern
            units="real",
            calibration=1.,
            include_direct_beam=False,  # seems to be a bug, use False to avoid an extra spot
        )
        spots, intensity, _ = sim._get_spots(**params)
        millers = np.round(
            sim.get_current_coordinates().coordinates
        ).astype(np.int16)
        return SimulatedPeaks(
            complex(0., 0.),
            to_complex(spots[:-1, ::-1]).ravel().conjugate(),
            xp.array(millers[:-1, ...]),
            xp.array(intensity[:-1, ...]),
        )

    def _dynamical_sim(
        self,
        experiment: ExperimentInformation,
        max_excitation_error: float | None = None,
        max_extent: float | None = None,
        lattice_mod: LatticeMultipliers = LatticeMultipliers(),
        backend: Literal["cupy", "cpu"] | types.ModuleType = "cpu",
    ):
        from .crystal_bloch import get_bloch_pattern, unpack_pattern

        xp, _ = get_backend(backend)
        if max_excitation_error is None:
            max_excitation_error = 0.1
        pattern = get_bloch_pattern(
            lattice_mod.apply_ase(self.atoms),
            self.orientation,
            progress=False,
            voltage=experiment.voltage_kv * 1_000,
            max_extent=max_extent,
            max_excitation_error=max_excitation_error,
            xp=xp
        )
        hkls, offsets, intensities = unpack_pattern(pattern, xp=xp)
        offsets.real *= -1
        return SimulatedPeaks(
            complex(0., 0.),
            offsets,
            np.asarray(hkls),
            intensities,
        )

    def peak_positions(
        self,
        experiment: ExperimentInformation,
        max_excitation_error: float | None = None,
        max_extent: float | None = None,
        lattice_mod: LatticeMultipliers = LatticeMultipliers(),
        dynamic_diff: bool = False,
        backend: Literal["cupy", "cpu"] | types.ModuleType = "cpu",
    ):
        if max_extent is None:
            max_extent = experiment.max_extent
        if dynamic_diff:
            fn = self._dynamical_sim
        else:
            fn = self._kinematical_sim
        peaks = fn(
            experiment,
            max_extent=max_extent,
            max_excitation_error=max_excitation_error,
            lattice_mod=lattice_mod,
            backend=backend,
        )
        return peaks.to_numpy()

    def synthetic(
        self,
        experiment: ExperimentInformation,
        sim_peaks: SimulatedPeaks,
        *,
        distortions: DistortionConfig = DistortionConfig(),
        frame_params: FrameParameters = FrameParameters(),
        backend: Literal["cupy", "cpu"] | types.ModuleType = "cpu",
    ):
        xp, ndimage = get_backend(backend)

        offsets = sim_peaks.peaks
        intensities = sim_peaks.weights
        pixel_peaks = sim_peaks.to_pixels(experiment)
        offsets = pixel_peaks.offsets
        warped = apply_distortion(offsets, distortions)
        frame = build_frame(
            experiment.frame_shape,
            experiment.pattern_centre_px,
            warped,
            experiment.radius_px,
            minor=experiment.ellipse_minor,
            orientation=experiment.ellipse_orientation,
            intensities=intensities,
            params=frame_params,
            xp=xp,
            ndimage=ndimage,
        )
        return frame
