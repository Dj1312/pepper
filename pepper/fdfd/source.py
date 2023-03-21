from enum import Enum
from pydantic import create_model

from tidy3d import GaussianPulse
import tidy3d.components.source as tidy3d_src

from ..constants import C_0
from .source_init import dict_src_init


DUMMY_WL = 1.55 * 1e-6
DUMMY_FREQ = C_0 / DUMMY_WL
DUMMY_FACTOR = 0.1
DUMMY_SRC_CLASS = GaussianPulse


class SimulationType(Enum):
    FDTD = "fdtd"
    FDFD = "fdfd"


# We need to use freq0 since autogrid calculations will use it
class DummySource(DUMMY_SRC_CLASS):
    def __init__(self, freq):
        super(DummySource, self).__init__(
            freq0=freq,
            fwidth=DUMMY_FACTOR * freq,
        )


def _tidy3d_fdfd_monkey_patch(Tidy3DClass):
    def fun_add_fields(freq0=None, wavelength=None, **kwds):
        if freq0 is None and wavelength is None:
            raise ValueError("Provide either 'freq0' or 'wavelength'.")
        elif freq0 is not None and wavelength is not None:
            raise ValueError("Provide only one field: 'freq0' or 'wavelength'.")
        elif freq0 is None:
            freq0 = C_0 / wavelength
        else:
            wavelength = C_0 / freq0

        additional_fdfd_fields = {
            'amplitude': 1.0,
            'phase': 0.0,
        }

        model = create_model(
            Tidy3DClass.__name__,
            __base__=Tidy3DClass,
            freq0=freq0,
            wavelength=wavelength,
            simulation_type=SimulationType.FDFD,
            source_initialization=dict_src_init[Tidy3DClass.__name__],
            **additional_fdfd_fields,
        )
        # used to solve the issue with the TypeError on 3.9.X
        # 'TypeError: Subscripted generics cannot be used with class and instance checks'
        return model(source_time=DummySource(freq0), **kwds)
    return fun_add_fields


PlaneWaveFdfd = _tidy3d_fdfd_monkey_patch(tidy3d_src.PlaneWave)
UniformCurrentSourceFdfd = _tidy3d_fdfd_monkey_patch(tidy3d_src.UniformCurrentSource)
PointDipoleFdfd = _tidy3d_fdfd_monkey_patch(tidy3d_src.PointDipole)
GaussianBeamFdfd = _tidy3d_fdfd_monkey_patch(tidy3d_src.GaussianBeam)
# AstigmaticGaussianBeamFdfd = _tidy3d_fdfd_monkey_patch(tidy3d_src.AstigmaticGaussianBeam)
AstigmaticGaussianBeamFdfd = _tidy3d_fdfd_monkey_patch(tidy3d_src.GaussianBeam)
ModeSourceFdfd = _tidy3d_fdfd_monkey_patch(tidy3d_src.ModeSource)
