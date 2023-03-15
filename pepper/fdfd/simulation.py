from abc import ABC
from enum import Enum
from functools import cached_property
from typing import Literal, Optional
from math import prod

import numpy as np
from pydantic import validator, root_validator, Extra


from tidy3d import Simulation as Tidy3dSim

from ..cache import cached_property
from ..constants import C_0


DUMMY_VALUE = -1


class SimulationType(Enum):
    FDTD = "fdtd"
    FDFD = "fdfd"


class SimulationFdfd(Tidy3dSim, extra=Extra.ignore):
    run_time: float = DUMMY_VALUE  # No runtime required in FDFD
    polarization: Literal['TE', 'TM']
    freq: Optional[float] = None
    wavelength: Optional[float] = None
    tfsf: bool = True

    src_inds: Optional[list] = None

    @root_validator(pre=False)
    def verify_freq(cls, values: dict):
        # Nothing is given
        # if list(values.values()).count(None) == 2:
        if values['freq'] is None and values['wavelength'] is None:
            raise ValueError("Provide either 'freq' or 'wavelength'.")
        # Both given
        if values['freq'] is not None and values['wavelength'] is not None:
            raise ValueError("Provide only one field: 'freq' or 'wavelength'.")

        if values['freq'] is None:
            values['freq'] = C_0 / values['wavelength']
        else:
            values['wavelength'] = C_0 / values['freq']

        return values

    def run(self, wavelength):
        if self.grid.num_cells.count(1) != 1:
            raise NotImplementedError("Actually, only 2D FDFD is supported!")

    @cached_property
    def eps(self):
        eps_temp = []
        for field in ['Ex', 'Ey', 'Ez']:
            eps = self.epsilon_on_grid(self.grid, coord_key=field).values
            # TODO: Make it 3D compatible
            eps_temp.append(eps.squeeze())
        return eps_temp

    # @cached_property
    # def source(self):
    #     arr_source = np.zeros(self.eps[0].shape, dtype=complex)
    #     for src in self.sources:
    #         print(src)
    #     return arr_source
