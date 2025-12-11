from typing import Callable

import numpy as np
from libertem.udf.base import UDF
from libertem.common.math import prod
from libertem.common.buffers import reshaped_view
from orix.quaternion import Rotation
import sparseconverter

from .crystal_orientation import ExperimentInformation, FrameParameters, OrientedPhase
from .frame_builder import xp, ndimage as xndimage

from scipy import ndimage


def make_frame(
        experiment, phase, frame_params,
        stretch_abc=(1., 1., 1.), scale_bc_ac_ab=(1., 1., 1.), rotate_deg=0.,
        max_excitation_error=None, bloch=False, xp=xp, ndimage=xndimage):
    sim_peaks = phase.peak_positions(
        experiment, stretch_abc=stretch_abc, scale_bc_ac_ab=scale_bc_ac_ab,
        rotate_deg=rotate_deg,
        max_excitation_error=max_excitation_error,
        bloch=bloch,
        xp=xp
    )
    # Play with the intensities to make the transmitted beam more similar to the diffracted beams
    #sim_peaks.modify_intensities(power=0.25)
    #sim_peaks.modify_000_intensity(multiply=2)
    sim_frame = phase.synthetic(
        experiment, sim_peaks, frame_params=frame_params, xp=xp, ndimage=ndimage
    )
    return sim_frame, sim_peaks


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
    def __init__(self, filename,
                 experiment: ExperimentInformation, phase: OrientedPhase,
                 frame_params: FrameParameters,
                 stretch_abc=(1., 1., 1.), scale_bc_ac_ab=(1., 1., 1.),
                 rotate_deg=0., max_excitation_error=None,
                 make_frame_fn: Callable = make_frame,
                 _is_master=True):
        # We keep a local copy that is not transferred to workers
        self._is_master = _is_master
        super().__init__(
            filename=filename,
            experiment=experiment,
            phase=phase,
            frame_params=frame_params,
            stretch_abc=stretch_abc,
            scale_bc_ac_ab=scale_bc_ac_ab,
            rotate_deg=rotate_deg,
            max_excitation_error=max_excitation_error,
            make_frame_fn=make_frame_fn,
            # This will be the value set on the worker nodes
            _is_master=False
        )

    def get_preferred_input_dtype(self):
        ''
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
        ''
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
        ''
        return {
            'peak_positions': self.buffer(kind='nav', dtype=object)
        }

    def get_task_data(self):
        ''
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
        ''
        rot = Rotation.from_euler(
            sparseconverter.for_backend(frame, sparseconverter.NUMPY)
        )
        p = self.params
        phase = p.phase.with_rot(rot)
        if self.xp.__name__ != 'cupy':
            xndimage = ndimage
        else:
            import cupyx.scipy.ndimage as xndimage
        sim_frame, sim_peaks = p.make_frame_fn(
            experiment=p.experiment,
            phase=phase,
            frame_params=p.frame_params,
            stretch_abc=p.stretch_abc,
            scale_bc_ac_ab=p.scale_bc_ac_ab,
            rotate_deg=p.rotate_deg,
            max_excitation_error=p.max_excitation_error,
            xp=self.xp,
            ndimage=xndimage,
        )
        sl = self.meta.slice.nav.get(self.task_data.memmap)
        sl[:] = self.forbuf(
            sim_frame,
            target=sl,
        )
        self.results.peak_positions[0] = sim_peaks.to_pixels(p.experiment)
