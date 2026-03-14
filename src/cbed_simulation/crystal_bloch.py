import numpy as np

from orix.quaternion.rotation import Rotation

from ase import Atoms
from abtem.bloch import BlochWaves, StructureFactor
from abtem.measurements import IndexedDiffractionPatterns


def get_bloch_pattern(
    atoms: Atoms,
    orientation: Rotation,
    progress: bool = False,
    voltage: float = 200e3,
    thickness_nm: float = 200,
    max_extent: float = 2.,
    max_excitation_error: float = 0.1,
    xp=np,
):
    if xp.__name__ == 'cupy':
        device = 'gpu'
    else:
        device = 'cpu'
    structure_factor = StructureFactor(
        atoms,
        # Parameter name changed recently, FIXME figure out how to support both
        # k_max=max_extent * 2,  # maximum scattering vector length (angle?)
        g_max=max_extent * 2,  # maximum scattering vector length (angle?)
        thermal_sigma=0.01,
        parametrization="lobato",
        device=device,
    )

    bloch_waves = BlochWaves(
        structure_factor=structure_factor,
        energy=voltage,
        sg_max=max_excitation_error,  # maximum excitation error,
        # Parameter name changed recently, FIXME figure out how to support both
        # k_max=max_extent,
        g_max=max_extent,
        orientation_matrix=orientation.to_matrix().squeeze(),
        device=device,
    )
    patterns = bloch_waves.calculate_diffraction_patterns(
        [thickness_nm * 10.],
    )
    return patterns.compute(progress_bar=progress)[0].to_cpu()


def unpack_pattern(
    patterns: IndexedDiffractionPatterns,
    intensity_threshold: float = 1e-10,
    xp=np,
):
    bloch_positions = tuple(
        complex(*position[:2]) for position, intensity
        in zip(patterns.positions, patterns.intensities)
        if intensity > intensity_threshold
    )
    bloch_intensity_dict = {
        tuple(hkl): intensity for hkl, intensity
        in zip(patterns.miller_indices, patterns.intensities)
        if intensity > intensity_threshold
    }

    spots = xp.asarray(bloch_positions)
    spots *= xp.exp(1j * np.pi)
    hkls = tuple(bloch_intensity_dict.keys())
    intensities = xp.asarray(
        tuple(bloch_intensity_dict.values())
    )
    return hkls, spots, intensities
