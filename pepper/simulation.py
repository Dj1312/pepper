from enum import Enum
from tidy3d import Simulation as Tidy3dSim


class SimulationType(Enum):
    FDTD = "fdtd"
    FDFD = "fdfd"


class Simulation(Tidy3dSim):
    
    def run(type: SimulationType, *args, **kwargs):
        if type == SimulationType.FDTD:
            raise NotImplementedError
        elif type == SimulationType.FDFD:
            print("Lessss go")
        else:
            raise ValueError