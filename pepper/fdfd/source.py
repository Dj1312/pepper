from enum import Enum
from pydantic import create_model

from tidy3d import GaussianPulse
import tidy3d.components.source as tidy3d_src

from ..constants import C_0


DUMMY_FACTOR = 0.1
DUMMY_SRC_CLASS = GaussianPulse


class SimulationType(Enum):
    FDTD = "fdtd"
    FDFD = "fdfd"


# TODO: Do we really need to use the freq0 for a source?
class DummySource(DUMMY_SRC_CLASS):
    def __init__(self, freq):
        super(DummySource, self).__init__(
            freq0=freq,
            fwidth=DUMMY_FACTOR * freq,
        )


# To pass pydantic checks, the source need to keep the same name
# A freq need to be provided for td.GridSpec.auto
def _tidy3d_fdfd_monkey_patch(Tidy3DClass):
    def fun_add_freq(sim_freq=None, sim_wl=None, **kwds):
        if sim_freq is None and sim_wl is None:
            raise ValueError("Provide either 'sim_freq' or 'sim_wl'.")
        elif sim_freq is not None and sim_wl is not None:
            raise ValueError("Provide only one field: 'sim_freq' or 'sim_wl'.")
        elif sim_freq is None:
            sim_freq = C_0 / sim_wl
        else:
            sim_wl = C_0 / sim_freq

        source_time = DummySource(sim_freq)
        simulation_type = SimulationType.FDFD
        model = create_model(
            Tidy3DClass.__name__,
            __base__=Tidy3DClass,
            sim_freq=sim_freq,
            sim_wl=sim_wl,
            source_time=source_time,
            simulation_type=simulation_type,
        )
        return model(**kwds)
    return fun_add_freq


UniformCurrentSourceFdfd = _tidy3d_fdfd_monkey_patch(tidy3d_src.UniformCurrentSource)
PointDipoleFdfd = _tidy3d_fdfd_monkey_patch(tidy3d_src.PointDipole)
GaussianBeamFdfd = _tidy3d_fdfd_monkey_patch(tidy3d_src.GaussianBeam)
AstigmaticGaussianBeamFdfd = _tidy3d_fdfd_monkey_patch(tidy3d_src.AstigmaticGaussianBeam)
ModeSourceFdfd = _tidy3d_fdfd_monkey_patch(tidy3d_src.ModeSource)
PlaneWaveFdfd = _tidy3d_fdfd_monkey_patch(tidy3d_src.PlaneWave)
