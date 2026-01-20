import numpy as np
from libertem.udf.base import UDF
from libertem.common.math import prod
from libertem.common.buffers import reshaped_view

from .crystal_orientation import ExperimentInformation, FrameParameters, OrientedPhase
from .utils import to_numpy


class CBEDSimUDF(UDF):
    '''
    Record input data as NumPy .npy file

    Parameters
    ----------

    filename : str or path-like
        Filename where to save. The file will be overwritten if it exists.
    _is_master : bool
        Internal flag, keep at default value.
    '''
    def __init__(
        self,
        filename,
        experiment: ExperimentInformation,
        phase: OrientedPhase,
        frame_params: FrameParameters,
        stretch_abc=(1., 1., 1.),
        scale_bc_ac_ab=(1., 1., 1.),
        max_excitation_error=None,
        _is_master=True,
    ):
        # We keep a local copy that is not transferred to workers
        self._is_master = _is_master
        super().__init__(
            filename=filename,
            experiment=experiment,
            phase=phase,
            frame_params=frame_params,
            stretch_abc=stretch_abc,
            scale_bc_ac_ab=scale_bc_ac_ab,
            max_excitation_error=max_excitation_error,
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
        if tuple(self.meta.dataset_shape.sig) != (3, ):
            raise ValueError(
                'This UDF expects a sig shape of (3, ) that corresponds to Euler angles, '
                f'received {self.meta.dataset_shape.sig} instead.'
            )
        if self.meta.input_dtype.kind != 'f':
            raise ValueError(
                'This UDF expects a floating point input dtype, '
                f'received {self.meta.input_dtype} instead.'
            )
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
        phase = phase.with_rot(
            to_numpy(frame),
        )
        sim_peaks = phase.peak_positions(
            p.experiment,
            stretch_abc=p.stretch_abc,
            scale_bc_ac_ab=p.scale_bc_ac_ab,
            max_excitation_error=p.max_excitation_error,
            bloch=False,
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
