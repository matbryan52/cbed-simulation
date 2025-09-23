import numpy as np
from typing import NamedTuple


class DistortionConfig(NamedTuple):
    spiral_strength: float = 0.
    elliptical_strength_parallel: float = 1.
    elliptical_strength_perpendicular: float = 1.
    elliptical_angle: float = 0.
    projective_strength_y: float = 0.
    projective_strength_x: float = 0.
    barrel_power: float = 0.


def spiral_warp(positions: np.ndarray, strength: float) -> np.ndarray:
    # strength in degrees / 100 px
    rad = np.abs(positions)
    angle = np.angle(positions, deg=True)
    angle = angle + (np.sqrt(rad) / 100.) * strength
    return rad * np.exp(1j * np.deg2rad(angle))


def elliptical_warp(
    positions: np.ndarray, para_str: float, perp_str: float, axis: float
) -> np.ndarray:
    # axis in degrees
    # strength in para, perp scale factors
    rad = np.abs(positions)
    angle = np.angle(positions)
    axis = np.deg2rad(axis)
    delta_angle = angle - axis
    perp = perp_str * rad * np.sin(delta_angle)
    para = para_str * rad * np.cos(delta_angle)
    return (para + perp * 1j) * np.exp(1j * axis)


def aligned_stretch(
    positions: np.ndarray, para_str: float, axis: float,
) -> np.ndarray:
    # axis in degrees
    # strength in para, perp scale factors
    rad = np.abs(positions)
    angle = np.angle(positions)
    axis = np.deg2rad(axis)
    delta_angle = angle - axis
    perp = rad * np.sin(delta_angle)
    para = para_str * rad * np.cos(delta_angle)
    return (para + perp * 1j) * np.exp(1j * axis)


def projective_warp(positions: np.ndarray, y_strength: float, x_strength: float):
    xx = positions.real
    yy = positions.imag
    xx_dst, yy_dst = xx + (y_strength * np.abs(yy)), yy + (x_strength * np.abs(xx))
    return xx_dst + yy_dst * 1j


def barrel_warp(positions: np.ndarray, power: float):
    rad = np.abs(positions)
    angle = np.angle(positions)
    return rad * (1 + power * rad) * np.exp(1j * angle)


def apply_distortion(positions: np.ndarray, params: DistortionConfig):
    positions = spiral_warp(positions, params.spiral_strength)
    positions = elliptical_warp(
        positions,
        params.elliptical_strength_parallel,
        params.elliptical_strength_perpendicular,
        params.elliptical_angle,
    )
    positions = projective_warp(
        positions,
        params.projective_strength_y,
        params.projective_strength_x,
    )
    positions = barrel_warp(positions, params.barrel_power)
    return positions
