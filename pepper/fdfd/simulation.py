from dataclasses import dataclass, asdict
# from pydantic.dataclasses import dataclass
from enum import Enum
from typing import Any, Literal, Optional, Tuple, Union

import numpy as np
from pydantic import root_validator, Extra


from tidy3d import Simulation as Tidy3dSim
import tidy3d.components.boundary as td_bnd

from .base import BaseSimulationFdfd, SimulationFdfd_TE, SimulationFdfd_TM
from ..cache import cached_property
from ..constants import PI


MICROMETERS = 1e-6
DUMMY_VALUE = -1
PMLLikeBoundary = [td_bnd.PML, td_bnd.StablePML, td_bnd.Absorber]


class SimulationType(Enum):
    FDTD = "fdtd"
    FDFD = "fdfd"


@dataclass
class ParamsHandlerFdfd:
    eps: np.ndarray = None
    dl: Tuple[float, ...] = ()
    npml: Tuple[int, ...] = ()
    omega: float = None
    wavelength: float = None
    bloch_conditions: Tuple[Union[float, None], ...] = ()


# The Handler will take care of the FDFD sim: params + Sim itself
class FdfdHandler:
    FdfdSim: Union[Any, BaseSimulationFdfd] = None
    params: ParamsHandlerFdfd = ParamsHandlerFdfd()


class SimulationFdfd(Tidy3dSim, extra=Extra.ignore):
    # User defined
    run_time: float = DUMMY_VALUE  # No runtime required in FDFD
    polarization: Literal['TE', 'TM']
    tfsf: bool = True

    # Internal
    handler: FdfdHandler = FdfdHandler()
    freq0: Optional[float] = None
    wavelength: Optional[float] = None

    @root_validator(pre=False)
    def verify_sources(cls, values: dict):
        wls = [src.wavelength for src in values.get('sources')]
        if len(wls) > 1 and not all(wl == wls[0] for wl in wls):
            raise ValueError("Not all sources are defined with the same "
                             + "frequency of wavelength.")
        freq0 = values.get('sources')[0].freq0
        values['freq0'] = freq0

        values['wavelength'] = wls[0]
        values['handler'].params.wavelength = wls[0]

        # # Source good -> Create sim_obj
        # omega = 2 * PI * C_0
        return values

    @root_validator(pre=False)
    def verify_bloch_conditions(cls, values: dict):
        bloch_phase = []
        for axis in ('xyz'):
            if isinstance(values.get('boundary_spec')[axis].plus, td_bnd.BlochBoundary):
                # Need to correct with a 2pi factor
                # -> Tidy3D express them with units of [1 / 2pi]
                # Cannot use .bloch_phase since FDFD receive only phi
                # TODO: Modify FDFD model to take full exp(1j * phi) ?
                bloch_phase.append(
                    2 * PI * values.get('boundary_spec')[axis].plus.bloch_vec
                )
            else:
                bloch_phase.append(None)
        if values['tfsf'] is True:
            bloch_phase = [-val if val is not None else None for val in bloch_phase]

        values['handler'].params.bloch_conditions = tuple(bloch_phase)
        return values

    # @root_validator(pre=False)
    # def verify_sim_obj(cls, values: dict):
    #     if values['polarization'] == 'TE':
    #         values['handler'].SimClass = SimulationFdfd_TE
    #     elif values['polarization'] == 'TM':
    #         values['handler'].SimClass = SimulationFdfd_TM
    #     return values

    def post_validation(self):
        params = self.handler.params

        # Define all the params
        params.eps = self.eps[-1].squeeze()

        params.dl = *(
            _grid.dl * MICROMETERS for _grid in [self.grid_spec.grid_x,self.grid_spec.grid_y]
        ),
        params.npml = *(
            self.boundary_spec[axis].minus.num_layers
            if any(isinstance(self.boundary_spec[axis].minus, bnd) for bnd in PMLLikeBoundary)
            else 0 for axis in 'xy'
        ),
        if self.polarization == 'TE':
            Sim = SimulationFdfd_TE
        elif self.polarization == 'TM':
            Sim = SimulationFdfd_TM

        self.handler.FdfdSim = Sim(**asdict(params))

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.post_validation()

    @cached_property
    def eps(self):
        eps_temp = []
        for field in ['Ex', 'Ey', 'Ez']:
            eps = self.epsilon_on_grid(self.grid, coord_key=field).values
            # TODO: Make it 3D compatible
            # eps_temp.append(eps.squeeze())
            eps_temp.append(eps)
        return eps_temp

    @cached_property
    def source(self):
        arr_source = []  # np.zeros(self.eps[0].shape, dtype=complex)
        for src in self.sources:
            src_initialized = src.source_initialization(self)
            if src_initialized.ndim > 2 and src_initialized.shape.count(1) != 1:
                raise ValueError("A source is not 2D.")
            arr_source.append(src_initialized)
            # arr_source += src.source_initialization(self)
        return arr_source

    # TODO Implement the use of eps_xx, eps_yy, eps_zz
    def run(self):
        if self.grid.num_cells.count(1) != 1:
            raise NotImplementedError("Actually, only 2D FDFD is supported.")
        src = np.sum(self.source, axis=0).flatten()
        return self.handler.FdfdSim.solve(src)
