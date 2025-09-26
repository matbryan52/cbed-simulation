import os
import copy
import pathlib
from typing import NamedTuple
import numpy as np
from orix.quaternion import Rotation
from orix.crystal_map import Phase
from orix.vector.miller import Miller
from orix.vector import Vector3d
from diffsims.generators.simulation_generator import SimulationGenerator
from diffsims.generators.zap_map_generator import get_rotation_from_z_to_direction
from diffpy.structure.parsers.p_cif import P_cif
from ase import Atoms
from ase.io import read as read_atoms

from .utils import to_complex, to_array
from .distortions import DistortionConfig, apply_distortion
from .crystal_bloch import scale_and_rotate, get_bloch_pattern, unpack_pattern
from .frame_builder import build_frame, FrameParameters


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
    transmitted_centre_px: complex
    radius_px: int
    pattern_scale_factor: float  # px / nm-1
    pattern_rotation: float = 0.  # degrees
    voltage_kv: float = 200.
    precession_angle: float = 1.

    @property
    def max_extent(self) -> float:
        br = complex(*self.frame_shape[::-1])
        tr = br.real + 0j
        bl = br.imag * 1j
        dist_px = max(
            abs(self.transmitted_centre_px),
            abs(br - self.transmitted_centre_px),
            abs(tr - self.transmitted_centre_px),
            abs(bl - self.transmitted_centre_px),
        )
        return dist_px / self.pattern_scale_factor

    @property
    def pixelsize(self) -> float:
        return 1 / self.pattern_scale_factor

    @property
    def spotsize(self) -> float:
        return self.radius_px * self.pixelsize

    def modify(self, **kwargs):
        params = self._asdict()
        params.update(kwargs)
        return type(self)(**params)


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
            spots=to_array(self.peaks)[:, ::-1],
            intensity=2,
            millers=self.hkls,
            scatter_alpha=point_alpha,
            interactive=interactive,
        )


class SimulatedPeaks(IndexedPeaks):
    def to_pixels(self, experiment: ExperimentInformation) -> IndexedPeaks:
        offsets = scale_and_rotate(
            self.offsets,
            experiment.pattern_scale_factor,
            experiment.pattern_rotation,
        )
        return IndexedPeaks(
            experiment.transmitted_centre_px,
            offsets,
            self.hkls,
            self.weights,
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
    phase: Phase
    atoms: Atoms
    orientation: Rotation

    @classmethod
    def from_cif(
        cls,
        cif_path: os.PathLike,
        orientation: tuple[int, int, int] | Rotation,
        pattern_rotation: float = 0.,
    ):
        cif_path = pathlib.Path(cif_path)
        with cif_path.open('r') as fp:
            cif_str = fp.read()
        cif = P_cif()
        structure = cif.parse(cif_str)
        spacegroup = cif.spacegroup
        phase = Phase(
            cif_path.stem,
            space_group=spacegroup,
            structure=copy.deepcopy(structure),
        )
        if (
            isinstance(orientation, (np.ndarray, tuple, list))
            and all(isinstance(v, int) for v in orientation)
        ):
            miller = Miller(hkl=orientation, phase=phase)
            orientation = get_rotation_from_z_to_direction(
                structure,
                miller.uvw.squeeze(),
            )
            orientation = Rotation.from_euler(
                orientation,
                degrees=True,
            )
            v_start = Vector3d.zvector()
            v_end = orientation * v_start
            in_plane = Rotation.from_axes_angles(
                v_end,
                pattern_rotation,
                degrees=True,
            )
            orientation = in_plane * orientation
        return cls(
            phase,
            read_atoms(cif_path),
            orientation,
        )

    def with_rot(self, orientation: Rotation | np.ndarray):
        if not isinstance(orientation, Rotation):
            orientation = Rotation.from_euler(orientation)
        return type(self)(
            self.phase,
            self.atoms,
            orientation,
        )

    def _kinematical_sim(
        self,
        experiment: ExperimentInformation,
        max_excitation_error: float = 0.03,
        max_extent: float | None = None,
    ):
        gen = SimulationGenerator(
            accelerating_voltage=experiment.voltage_kv,
            precession_angle=experiment.precession_angle,
        )
        sim = gen.calculate_diffraction2d(
            phase=self.phase,
            rotation=self.orientation,
            reciprocal_radius=max_extent,
            max_excitation_error=max_excitation_error,
            with_direct_beam=True,
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
            to_complex(spots[:-1, ::-1]).ravel(),
            millers[:-1, ...],
            intensity[:-1, ...],
        )

    def _dynamical_sim(
        self,
        experiment: ExperimentInformation,
        max_excitation_error: float = 0.1,
        max_extent: float | None = None,
        stretch_abc: tuple[float, float, float] = (1., 1., 1.),
        scale_bc_ac_ab: tuple[float, float, float] = (1., 1., 1.),
    ):
        pattern = get_bloch_pattern(
            self.atoms,
            self.orientation.inv(),
            progress=False,
            stretch_abc=stretch_abc,
            scale_bc_ac_ab=scale_bc_ac_ab,
            voltage=experiment.voltage_kv * 1_000,
            max_extent=max_extent,
            max_excitation_error=max_excitation_error,
        )
        hkls, offsets, intensities = unpack_pattern(pattern)
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
        stretch_abc: tuple[float, float, float] = (1., 1., 1.),
        scale_bc_ac_ab: tuple[float, float, float] = (1., 1., 1.),
        rotate_deg: float = 0.,
        bloch: bool = True,
    ):
        if max_extent is None:
            max_extent = experiment.max_extent
        fn = self._kinematical_sim
        if bloch:
            fn = self._dynamical_sim
        kwargs = dict(
            max_extent=max_extent,
        )
        if not bloch and (stretch_abc != (1., 1., 1.) or scale_bc_ac_ab != (1., 1., 1.)):
            raise NotImplementedError("No support for strained crystal with kinematic simulation")
        else:
            kwargs["stretch_abc"] = stretch_abc
            kwargs["scale_bc_ac_ab"] = scale_bc_ac_ab
        if max_excitation_error is not None:
            kwargs["max_excitation_error"] = max_excitation_error
        peaks = fn(
            experiment,
            **kwargs,
        )
        peaks.offsets[:] *= np.exp(1j * np.deg2rad(rotate_deg))
        return peaks

    def synthetic(
        self,
        experiment: ExperimentInformation,
        sim_peaks: SimulatedPeaks,
        distortions: DistortionConfig = DistortionConfig(),
        frame_params: FrameParameters = FrameParameters(),
    ):
        offsets = sim_peaks.peaks
        intensities = sim_peaks.weights
        pixel_peaks = sim_peaks.to_pixels(experiment)
        offsets = pixel_peaks.offsets
        warped = apply_distortion(offsets, distortions)
        frame = build_frame(
            experiment.frame_shape,
            experiment.transmitted_centre_px,
            warped,
            experiment.radius_px,
            intensities=intensities,
            params=frame_params,
        )
        return frame
