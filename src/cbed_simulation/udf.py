import os
import numpy as np
import libertem.api as lt
from libertem.udf.base import UDF
from libertem.common.math import prod
from libertem.common.buffers import reshaped_view

from .crystal_orientation import (
    ExperimentInformation, FrameParameters, OrientedPhase, LatticeMultipliers
)


class CBEDSimUDF(UDF):
    '''
    Build CBED frames into a .npy file

    Accepts AUX data for orientation, stretch and scale
    '''
    def __init__(
        self,
        filename: os.PathLike,
        experiment: ExperimentInformation,
        phase: OrientedPhase,
        frame_params: FrameParameters,
        orientation: tuple[float, float, float] | None = None,
        lattice_mod: LatticeMultipliers = LatticeMultipliers(),
        max_excitation_error=None,
        dynamic_diff: bool = False,
        _is_master=True,
    ):
        # We keep a local copy that is not transferred to workers
        self._is_master = _is_master
        super().__init__(
            filename=filename,
            experiment=experiment,
            phase=phase,
            frame_params=frame_params,
            orientation=orientation,
            lattice_mod=lattice_mod,
            max_excitation_error=max_excitation_error,
            dynamic_diff=dynamic_diff,
            # This will be the value set on the worker nodes
            _is_master=False
        )

    def get_preferred_input_dtype(self):
        return self.USE_NATIVE_DTYPE

    @property
    def output_shape(self):
        return tuple(self.meta.dataset_shape.nav) + self.params.experiment.frame_shape

    @property
    def flat_output_shape(self):
        return (prod(self.meta.dataset_shape.nav), ) + self.params.experiment.frame_shape

    @property
    def output_dtype(self):
        return int

    def preprocess(self):
        # create the file once in the preprocess method on the master node
        if self._is_master:
            np.lib.format.open_memmap(
                self.params.filename,
                mode='w+',
                dtype=self.output_dtype,
                shape=self.output_shape,
            )

    def get_result_buffers(self):
        return {
            'peak_positions': self.buffer(kind='nav', dtype=object)
        }

    def get_task_data(self):
        m = np.lib.format.open_memmap(
                self.params.filename,
                mode='r+',
                dtype=self.output_dtype,
                shape=self.output_shape,
        )
        return {
            'memmap': reshaped_view(m, self.flat_output_shape)
        }

    def get_backends(self):
        return (self.BACKEND_CUPY, self.BACKEND_NUMPY)

    def process_frame(self, frame):
        p = self.params
        phase: OrientedPhase = p.phase
        if p.orientation is not None:
            phase = phase.with_rot(
                p.orientation,
            )
        if isinstance(p.lattice_mod, LatticeMultipliers):
            lattice_mod = p.lattice_mod
        else:
            # aux_data case: lattice_mod is in an array
            lattice_mod = p.lattice_mod.item()
        sim_peaks = phase.peak_positions(
            p.experiment,
            max_excitation_error=p.max_excitation_error,
            lattice_mod=lattice_mod,
            dynamic_diff=p.dynamic_diff,
            backend=self.xp,
        )
        sim_frame = phase.synthetic(
            p.experiment,
            sim_peaks,
            frame_params=p.frame_params,
            backend=self.xp,
        )
        sl = self.meta.slice.nav.get(
            self.task_data.memmap
        )
        sl[:] = self.forbuf(
            sim_frame,
            target=sl,
        )
        self.results.peak_positions[0] = sim_peaks.to_pixels(
            p.experiment,
            clip=True,
        )


def build_udf_ds(
    out_path: os.PathLike,
    nav_shape: tuple[int, int],
    ctx: lt.Context,
    phase: OrientedPhase,
    experiment: ExperimentInformation,
    frame_parameters: FrameParameters,
    orientation: tuple[float, float, float] | np.ndarray | None = None,
    lattice_mod: LatticeMultipliers | np.ndarray = None,
    max_excitation_error=None,
    dynamic_diff: bool = False,
):
    num_frames = np.prod(nav_shape, dtype=int)
    ds = ctx.load("memory", data=np.arange(num_frames).reshape(*nav_shape, 1, 1))
    if lattice_mod is None:
        lattice_mod = LatticeMultipliers()
    elif isinstance(lattice_mod, np.ndarray):
        assert lattice_mod.shape == nav_shape, "lattice_mod array incompatible with aux_data"
        lattice_mod = CBEDSimUDF.aux_data(
            lattice_mod.ravel(), kind="nav", dtype=object
        )
    else:
        raise ValueError("Unsupported lattice_mod type")
    # NOTE initial orientation of phase will be ignored
    if orientation is not None:
        orientation = np.asarray(orientation)
        if orientation.size != 3:
            assert orientation.shape == nav_shape + (3,), "orientation incompatible with aux_data"
            orientation = CBEDSimUDF.aux_data(
                orientation.ravel(), kind="nav", extra_shape=(3,)
            )
    udf = CBEDSimUDF(
        out_path,
        experiment,
        phase,
        frame_parameters,
        lattice_mod=lattice_mod,
        orientation=orientation,
        max_excitation_error=max_excitation_error,
        dynamic_diff=dynamic_diff,
    )
    return udf, ds
