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
    def fun_add_fields(sim_freq=None, sim_wl=None, **kwds):
        if sim_freq is None and sim_wl is None:
            raise ValueError("Provide either 'sim_freq' or 'sim_wl'.")
        elif sim_freq is not None and sim_wl is not None:
            raise ValueError("Provide only one field: 'sim_freq' or 'sim_wl'.")
        elif sim_freq is None:
            sim_freq = C_0 / sim_wl
        else:
            sim_wl = C_0 / sim_freq

        additional_fdfd_fields = {
            'amplitude': 1.0,
            'phase': 0.0,
        }

        model = create_model(
            Tidy3DClass.__name__,
            __base__=Tidy3DClass,
            sim_freq=sim_freq,
            sim_wl=sim_wl,
            source_time=DummySource(sim_freq),
            simulation_type=SimulationType.FDFD,
            source_initialization=dict_src_init,  # [Tidy3DClass.__name__],
            **additional_fdfd_fields,
        )
        return model(**kwds)
    return fun_add_fields


PlaneWaveFdfd = _tidy3d_fdfd_monkey_patch(tidy3d_src.PlaneWave)
UniformCurrentSourceFdfd = _tidy3d_fdfd_monkey_patch(tidy3d_src.UniformCurrentSource)
PointDipoleFdfd = _tidy3d_fdfd_monkey_patch(tidy3d_src.PointDipole)
GaussianBeamFdfd = _tidy3d_fdfd_monkey_patch(tidy3d_src.GaussianBeam)
# AstigmaticGaussianBeamFdfd = _tidy3d_fdfd_monkey_patch(tidy3d_src.AstigmaticGaussianBeam)
AstigmaticGaussianBeamFdfd = _tidy3d_fdfd_monkey_patch(tidy3d_src.GaussianBeam)
ModeSourceFdfd = _tidy3d_fdfd_monkey_patch(tidy3d_src.ModeSource)
