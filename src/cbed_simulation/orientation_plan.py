import numpy as np
from scipy import constants
from typing import Literal, NamedTuple
from scipy.spatial.transform import Rotation as RotationSP
from .utils import to_numpy


def electron_wavelength_angstrom(E_eV: float) -> float:
    m = constants.m_e
    e = constants.elementary_charge
    c = constants.c
    h = constants.h
    return h / np.sqrt(2 * m * e * E_eV) / np.sqrt(1 + e * E_eV / 2 / m / c**2) * 10**10


def get_num_angle_steps(vector_range, angle_step: float):
    # Solve for number of angular steps in zone axis (rads)
    angle_u_v = np.arccos(
        np.sum(
            vector_range[0, :]
            * vector_range[1, :]
        )
    )
    angle_u_w = np.arccos(
        np.sum(
            vector_range[0, :]
            * vector_range[2, :]
        )
    )
    max_num_steps = np.round(
        np.maximum(
            np.rad2deg(angle_u_v) / angle_step,
            np.rad2deg(angle_u_w) / angle_step,
        )
    ).astype(int)
    return max_num_steps, angle_u_v, angle_u_w


def _generate_points_slerp(zone_axis_steps, zone_axis_range, angle_u_v, angle_u_w):
    # Generate points spanning the zone axis range
    # Calculate points along u and v using the SLERP formula
    # https://en.wikipedia.org/wiki/Slerp
    weights = np.linspace(0, 1, zone_axis_steps + 1)
    pv = zone_axis_range[0, :] * np.sin(
        (1 - weights[:, None]) * angle_u_v
    ) / np.sin(angle_u_v) + zone_axis_range[1, :] * np.sin(
        weights[:, None] * angle_u_v
    ) / np.sin(
        angle_u_v
    )

    # Calculate points along u and w using the SLERP formula
    pw = zone_axis_range[0, :] * np.sin(
        (1 - weights[:, None]) * angle_u_w
    ) / np.sin(angle_u_w) + zone_axis_range[2, :] * np.sin(
        weights[:, None] * angle_u_w
    ) / np.sin(
        angle_u_w
    )
    return pv, pw


def gen_points_slerp_2(num_zones, zone_axis_steps, zone_axis_range, pv, pw):
    vecs = np.zeros((num_zones, 3))
    vecs[0, :] = zone_axis_range[0, :]
    inds = np.zeros((num_zones, 3), dtype="int")

    # Calculate zone axis points on the unit sphere with another application of SLERP
    for a0 in np.arange(1, zone_axis_steps + 1):
        a0_inds = np.arange(
            a0 * (a0 + 1) / 2, a0 * (a0 + 1) / 2 + a0 + 1
        ).astype(
            int
        )

        p0 = pv[a0, :]
        p1 = pw[a0, :]

        weights = np.linspace(0, 1, a0 + 1)

        angle_p = np.arccos(np.sum(p0 * p1))

        vecs[a0_inds, :] = p0[None, :] * np.sin(
            (1 - weights[:, None]) * angle_p
        ) / np.sin(angle_p) + p1[None, :] * np.sin(
            weights[:, None] * angle_p
        ) / np.sin(
            angle_p
        )

        inds[a0_inds, 0] = a0
        inds[a0_inds, 1] = np.arange(a0 + 1)
    return vecs


def gen_new_vecs(vecs, transform, tol_distance):
    vec_new = np.copy(vecs) * np.array(transform)
    keep = np.zeros(vec_new.shape[0], dtype="bool")
    for a0 in range(keep.size):
        if (
            np.sqrt(
                np.min(
                    np.sum((vecs - vec_new[a0, :]) ** 2, axis=1)
                )
            )
            > tol_distance
        ):
            keep[a0] = True

    vecs = np.vstack((vecs, vec_new[keep, :]))
    return vecs


def get_azim_elev(a: float | np.ndarray, b: float | np.ndarray, c: float | np.ndarray):
    azim = np.arctan2(a, b)
    elev = np.arctan2(
        np.hypot(a, b),
        c,
    )
    return azim, elev


def vec_for_azim_elev(azim, elev):
    s_a = np.sin(azim)
    c_a = np.cos(azim)
    s_e = np.sin(elev)
    c_e = np.cos(elev)
    return np.asarray([
        s_e * s_a,
        s_e * c_a,
        c_e,
    ])


def _rot_matrix(theta, mult=1):
    return np.stack(
        [
            np.stack([np.cos(theta), mult * np.sin(theta)], axis=-1),
            np.stack([-np.sin(theta) * mult, np.cos(theta)], axis=-1),
        ],
        axis=-2,
    )


def get_rotation_matrix(
    a: float | np.ndarray, b: float | np.ndarray, c: float | np.ndarray
) -> np.ndarray:
    azim, elev = get_azim_elev(a, b, c)
    azim_array = _rot_matrix(azim)
    elev_array = _rot_matrix(elev)
    mult = np.rot90(np.eye(2)) * -1
    num_el = np.asarray(a).size
    m1z = np.zeros((num_el, 3, 3), dtype=float)
    m1z[:, 0, 0] = 1
    m1z[:, -1, -1] = 1
    m2x = m1z.copy()
    m3z = m1z.copy()
    m1z[:, :2, :2] = azim_array[np.newaxis, ...]
    m2x[:, -2:, -2:] = elev_array[np.newaxis, ...]
    m3z[:, :2, :2] = azim_array[np.newaxis, ...] * mult[np.newaxis, ...]
    rotation_matrix = m1z @ m2x @ m3z
    # m1z = np.array(
    #     [
    #         [np.cos(azim), np.sin(azim), 0],
    #         [-np.sin(azim), np.cos(azim), 0],
    #         [0, 0, 1],
    #     ]
    # )
    # m2x = np.array(
    #     [
    #         [1, 0, 0],
    #         [0, np.cos(elev), np.sin(elev)],
    #         [0, -np.sin(elev), np.cos(elev)],
    #     ]
    # )
    # m3z = np.array(
    #     [
    #         [np.cos(azim), -np.sin(azim), 0],
    #         [np.sin(azim), np.cos(azim), 0],
    #         [0, 0, 1],
    #     ]
    # )
    if np.asarray(a).shape == tuple():
        return rotation_matrix.squeeze()
    return rotation_matrix


def calculate_rotation_matrices(vecs):
    return get_rotation_matrix(vecs[:, 0], vecs[:, 1], vecs[:, 2])


def get_in_plane_steps(
    step_deg: float = 2.,
):
    # Solve for number of angular steps along in-plane rotation direction
    in_plane_steps = np.round(
        360 / step_deg
    ).astype(
        int
    )
    # Calculate -z angles (Euler angle 3)
    return np.linspace(
        0, 2 * np.pi, num=in_plane_steps, endpoint=False
    )


def orientation_plan(
    angle_step: float = 2.,
    tol_distance: float = 0.01,
    expand_to: Literal["quarter", "half"] | None = "half",
    zone_axis_range: np.ndarray | None = None,
):
    assert expand_to in (None, "quarter", "half")

    if zone_axis_range is None:
        zone_axis_range = np.array(
            [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
        )
    num_steps, angle_u_v, angle_u_w = get_num_angle_steps(
        zone_axis_range, angle_step
    )
    num_zones = (
        (num_steps + 1)
        * (num_steps + 2)
        / 2
    ).astype(int)

    pv, pw = _generate_points_slerp(
        num_steps, zone_axis_range, angle_u_v, angle_u_w
    )
    vecs = gen_points_slerp_2(
        num_zones, num_steps, zone_axis_range, pv, pw
    )

    if expand_to is not None and expand_to in ("quarter", "half"):
        # expand to quarter sphere
        vecs = gen_new_vecs(vecs, (-1, 1, 1), tol_distance)
    if expand_to is not None and expand_to in ("half"):
        # expand to hemisphere
        vecs = gen_new_vecs(vecs, (1, -1, 1), tol_distance)

    return vecs


class CorrelationParameters(NamedTuple):
    tol_distance: float = 0.01
    sigma_excitation_error: float = 0.02
    precession_angle: float | None = None
    kernel_size: float = 0.08
    power_intensity: float = 0.25
    power_intensity_experiment: float = 0.25
    power_radial: float = 1.0


class PolarMapping(NamedTuple):
    radii: np.ndarray
    count: np.ndarray
    index: np.ndarray
    gamma: np.ndarray
    corr_params: CorrelationParameters

    @property
    def size(self):
        return self.radii.size

    @property
    def in_plane_steps(self):
        return self.gamma.size


class StructureFactor(NamedTuple):
    g_vecs: np.ndarray
    intensities: np.ndarray
    wavelength: float


def get_corr_polar_mapping(
    g_vec_all: np.ndarray,
    in_plane_step: float,
    correlation_params: CorrelationParameters,
):
    p = correlation_params
    tol_distance = p.tol_distance
    # g_vec_all.shape == (3, n)
    g_vec_leng = np.linalg.norm(g_vec_all, axis=0)

    # Determine the radii of all spherical shells in the structure factor
    radii_test = np.round(g_vec_leng / tol_distance) * tol_distance
    radii = np.unique(radii_test)
    shell_radii = radii[np.abs(radii) > tol_distance]  # Remove zero beam
    shell_count = np.zeros(shell_radii.size)
    shell_index = -1 * np.ones(
        g_vec_all.shape[1], dtype=int
    )

    # Assign each structure factor point to a radial shell
    for a0 in range(shell_radii.size):
        sub = (
            np.abs(shell_radii[a0] - radii_test)
            <= tol_distance / 2
        )
        shell_index[sub] = a0
        shell_count[a0] = np.sum(sub)
        shell_radii[a0] = np.mean(g_vec_leng[sub])
    return PolarMapping(
        shell_radii,
        shell_count,
        shell_index,
        get_in_plane_steps(in_plane_step),
        correlation_params,
    )


def excitation_errors(
    g,
    wavelength,
    precession_angle_degrees=None,
    precession_steps=72,
):
    """
    Calculate the excitation errors, assuming k0 = [0, 0, -1/lambda].
    If foil normal is not specified, we assume it is [0,0,-1].

    Precession is currently implemented using numerical integration.
    """

    if precession_angle_degrees is None:
        return (2 * g[2, :] - wavelength * np.sum(g * g, axis=0)) / (
            2 - 2 * wavelength * g[2, :]
        )

    else:
        t = np.deg2rad(precession_angle_degrees)
        p = np.linspace(
            0,
            2.0 * np.pi,
            precession_steps,
            endpoint=False,
        )
        foil_normal = np.array((0.0, 0.0, -1.0))

        k = np.reshape(
            (-1 / wavelength)
            * np.vstack(
                (
                    np.sin(t) * np.cos(p),
                    np.sin(t) * np.sin(p),
                    np.cos(t) * np.ones(p.size),
                )
            ),
            (3, 1, p.size),
        )

        term1 = np.sum((g[:, :, None] + k) * foil_normal[:, None, None], axis=0)
        term2 = np.sum((g[:, :, None] + 2 * k) * g[:, :, None], axis=0)
        sg = np.sqrt(term1**2 - term2) - term1

        return sg


def correlogram_for_orientation(
    structure_factor: StructureFactor,
    rotation_matrix: np.ndarray,
    polar_mapping: PolarMapping,
    sf: np.ndarray | None = None,
):
    """
    Maps the theoretical diffraction pattern for this orientation
    into an (r, γ) projection, via interpolation of the
    intensities into bins defined by shells and gamma
    """
    p = polar_mapping.corr_params
    gamma = polar_mapping.gamma
    g = rotation_matrix.T @ structure_factor.g_vecs
    sg = excitation_errors(g, structure_factor.wavelength)  # , precession_angle_degrees=...)

    # Keep only points that will contribute to this orientation plan slice
    keep = np.logical_and(
        np.abs(sg) < p.kernel_size,
        polar_mapping.index >= 0,
    )

    # calculate intensity of spots
    if p.precession_angle is None:
        Ig = np.exp(sg[keep] ** 2 / (-2 * p.sigma_excitation_error**2))
    else:
        # precession extension
        prec = np.cos(np.linspace(0, 2 * np.pi, 90, endpoint=False))
        dsg = np.tan(p.precession_angle) * np.sum(
            g[:2, keep] ** 2, axis=0
        )
        Ig = np.mean(
            np.exp(
                (sg[keep, None] + dsg[:, None] * prec[None, :]) ** 2
                / (-2 * p.sigma_excitation_error**2)
            ),
            axis=1,
        )

    # in-plane rotation angle
    phi = np.arctan2(g[1, keep], g[0, keep])
    assert gamma[0] == 0.
    gamma_step = gamma[1]
    phi_ind = phi / gamma_step
    phi_floor = np.floor(phi_ind).astype(int)  # index in gamma of each spot's phi
    dphi = phi_ind - phi_floor  # delta gamma betwwen phi_floor[int] and next phi index
    # phi_floor, dphi = np.divmod(phi, gamma_step)
    # phi_floor = phi_floor.astype(int)

    # write intensities into orientation plan slice
    if sf is None:
        sf = np.zeros((polar_mapping.radii.size, gamma.size), dtype=float)
    radial_inds = polar_mapping.index[keep]
    multiplier = (
        ((structure_factor.intensities[keep] * Ig) ** p.power_intensity)
        * (polar_mapping.radii[radial_inds] ** p.power_radial)
    )

    # This interpolates the intensity of each spot into the gamma bins
    sf[
        radial_inds, phi_floor
    ] += (1 - dphi) * multiplier  # (1-dphi) of the intensity goes into phi_floor
    sf[
        radial_inds, np.mod(phi_floor + 1, gamma.size)
    ] += dphi * multiplier   # (dphi) of the intensity goes into (phi_floor + 1)

    # normalization
    orientation_ref_norm = np.linalg.norm(sf)
    if orientation_ref_norm > 0:
        sf /= orientation_ref_norm

    return sf


def braggpeaks_to_correlogram(
    qx: np.ndarray,
    qy: np.ndarray,
    intensity: np.ndarray,
    polar_mapping: PolarMapping,
):
    """
    Maps the braggpeaks into a (r-γ) correlogram defined by polar_mapping
    """
    p = polar_mapping.corr_params
    gamma = polar_mapping.gamma

    # Convert Bragg peaks to polar coordinates
    qr = np.sqrt(qx ** 2 + qy ** 2)
    qphi = np.arctan2(qy, qx)  # FIXME technically this is the wrong usage of arctan2

    # Calculate polar Bragg peak image
    im_sf = np.zeros(
        (
            polar_mapping.radii.size,
            polar_mapping.in_plane_steps,
        ),
        dtype=float,
    )

    for ind_radial, radius in enumerate(polar_mapping.radii):
        dqr = np.abs(qr - radius)
        sub = dqr < p.kernel_size  # mask of peaks within this radial shell

        if not sub.any():
            continue

        _intensity = np.power(
            np.maximum(intensity[sub, None], 0.0),
            p.power_intensity_experiment,
        )
        _gamma = np.mod(
            gamma[None, :] - qphi[sub, None] + np.pi,
            2 * np.pi,
        )
        _gamma -= np.pi

        _exponent = (
            (dqr[sub, None] ** 2 + (_gamma * radius) ** 2)
            / (-2 * p.kernel_size**2)
        )

        im_sf[ind_radial, :] = np.sum(
            _intensity * np.exp(_exponent),
            axis=0,  # sum over peaks in this radial shell, for all gammas
        )
    return im_sf


def _best_fit_orientation(
    sf_fft_conj: np.ndarray,
    im_sf_fft: np.ndarray,
    inverse: bool,
    xp=np,
):
    if inverse:
        im_sf_fft = im_sf_fft.conj()
    corr = xp.sum(
        xp.fft.ifft(
            sf_fft_conj * im_sf_fft
        ).real,
        axis=-2,  # sum over shells
    )
    best_corr = xp.argmax(corr).item()  # argmax over gamma

    or_idx, gamma_idx = np.unravel_index(best_corr, corr.shape)
    inds = np.arange(gamma_idx - 1, gamma_idx + 2) % corr.shape[-1]
    c = to_numpy(corr[or_idx, inds])
    gamma_corr_weight = (c[2] - c[0]) / (4 * c[1] - 2 * c[0] - 2 * c[2])

    corr_value = c[1]
    return (or_idx, gamma_idx), corr_value, gamma_corr_weight


def optimal_rotation_matrix(
    best_idcs: tuple[int, int],
    rotation_matrices: np.ndarray,
    gamma: np.ndarray,
    corr_gamma_weight: float,
    inverse_match: bool,
):
    # apply in-plane rotation, and inversion if needed
    or_idx, gamma_idx = best_idcs
    phi = gamma[gamma_idx]
    dphi = gamma[1] - gamma[0]
    phi += corr_gamma_weight * dphi
    if inverse_match:
        phi += np.pi
    m3z = np.array(
        [
            [np.cos(phi), np.sin(phi), 0],
            [-np.sin(phi), np.cos(phi), 0],
            [0, 0, 1],
        ]
    )
    orientation_matrix = rotation_matrices[or_idx] @ m3z
    if inverse_match:
        # Rotate 180 degrees around x axis for projected x-mirroring operation
        orientation_matrix[:, 1:] = -orientation_matrix[:, 1:]
    return orientation_matrix


def best_fit_orientation(
    sf_fft_conj: np.ndarray,
    im_sf_fft: np.ndarray,
    rotation_matrices: np.ndarray,
    polar_mapping: PolarMapping,
    check_inverse: bool = True,
    xp=np,
):
    best_idcs, corr_value, gamma_corr_weight = _best_fit_orientation(
        sf_fft_conj, im_sf_fft, inverse=False, xp=xp
    )
    inverse_match = False
    if check_inverse:
        best_idcs_i, corr_value_i, gamma_corr_weight_i = _best_fit_orientation(
            sf_fft_conj, im_sf_fft, inverse=True, xp=xp
        )
        if corr_value_i > corr_value:
            inverse_match = True
            best_idcs = best_idcs_i
            corr_value = corr_value_i
            gamma_corr_weight = gamma_corr_weight_i
    match_orientation_matrix = optimal_rotation_matrix(
        best_idcs,
        rotation_matrices,
        polar_mapping.gamma,
        gamma_corr_weight,
        inverse_match=inverse_match,
    )
    return match_orientation_matrix, corr_value, best_idcs, inverse_match


def py4dstem_to_euler(matrix: np.ndarray) -> tuple[float, float, float]:
    angles = RotationSP.from_matrix(matrix).as_euler("zxz")
    angles[0] -= np.pi/2
    angles[0] *= -1
    angles[2] *= -1
    angles = np.rad2deg(angles)
    return tuple(angles)


def compute_oriented_correlograms(
    rotation_matrices: np.ndarray,
    polar_mapping: PolarMapping,
    structure_factor: StructureFactor,
):
    sfs = np.zeros((
        rotation_matrices.shape[0],
        polar_mapping.size,
        polar_mapping.in_plane_steps,
    ), dtype=float)
    for ind, or_matrix in enumerate(rotation_matrices):
        correlogram_for_orientation(
            structure_factor,
            or_matrix,
            polar_mapping,
            sf=sfs[ind],
        )
    return sfs


# def refine_orientation(
#     im_sf_fft: np.ndarray,
#     start_vec: np.ndarray,
#     dTheta: float,
#     num_steps: int,
#     shells: PolarMapping,
#     gamma_step: np.ndarray,
#     g_vec_all: np.ndarray,
#     struct_factors_int: np.ndarray,
#     wavelength: float,
#     xp=np,
# ):
#     azim, elev = get_azim_elev(*start_vec)
#     azim_range = np.linspace(
#         np.deg2rad(-dTheta),
#         np.deg2rad(dTheta),
#         num=num_steps,
#         endpoint=True,
#     )
#     elev_range = azim_range.copy()
#     dda, dde = np.meshgrid(azim_range, elev_range)
#     refine_vecs = tuple(
#         vec_for_azim_elev(azim + da, elev + de)
#         for da, de
#         in zip(dda.ravel(), dde.ravel())
#     )
#     refine_vecs = np.stack(refine_vecs, axis=0)
#     rotation_matrices = calculate_rotation_matrices(refine_vecs)

#     sfs = compute_oriented_structure_factors(
#         rotation_matrices,
#         shells,
#         gamma,
#         g_vec_all,
#         struct_factors_int,
#         wavelength,
#     )

#     sf_fft_conj = xp.fft.fft(sfs).conj()
#     return best_fit_orientation(
#         sf_fft_conj,
#         im_sf_fft,
#         rotation_matrices,
#         gamma,
#         xp=xp,
#     )
