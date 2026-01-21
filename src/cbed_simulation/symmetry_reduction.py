import numpy as np
from orix.quaternion import Orientation


def symmetry_reduction(
    angles: tuple[float, float, float],
    point_group: int | str,
    degrees: bool | None = False,
):
    """
    Get symmetry reduced Euler angles based on a crystal's point group

    Untested!
    """
    orientation = Orientation.from_euler(
        euler=angles,
        symmetry=point_group,
        degrees=True,
        direction='lab2crystal',
    )
    pg = orientation.symmetry.proper_subgroup
    # get symmetry equivalent rotations from subgroup
    orient = pg._special_rotation.outer(orientation)
    alpha, beta, gamma = np.split(
        orient.to_euler().squeeze(), 3, axis=1
    )
    alpha = alpha.squeeze()
    beta = beta.squeeze()
    gamma = gamma.squeeze()

    # reduce 3rd euler angle by the primary axis order
    gamma = np.mod(gamma, 2 * np.pi / pg._primary_axis_order)
    # get the fundamental region
    max_alpha, max_beta, max_gamma = np.deg2rad(pg.euler_fundamental_region)
    # select euler angle sets inside the fundamental region
    is_inside = (alpha <= max_alpha) & (beta <= max_beta) & (gamma <= max_gamma)
    valid_euler = np.stack(
        (alpha[is_inside], beta[is_inside], gamma[is_inside]),
        axis=1,
    )
    # find the set of angles with the smallest sum
    min_row = np.argmin(valid_euler.sum(axis=1))
    if degrees:
        return np.rad2deg(valid_euler[min_row])
    return valid_euler[min_row]
