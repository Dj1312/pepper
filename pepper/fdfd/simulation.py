from abc import ABC
from enum import Enum
from typing import Literal, Optional
from functools import cached_property
from math import prod

import numpy as np
from pydantic import validator, root_validator, Extra


from tidy3d import Simulation as Tidy3dSim

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

    @root_validator(pre=False)
    def verify_freq(cls, values: dict):
        # Nothing is given
        if list(values.values()).count(None) == 2:
            raise ValueError("Provide either 'freq' or 'wavelength'.")
        # Both given
        elif list(values.values()).count(None) == 0:
            raise ValueError("Provide only one field: 'freq' or 'wavelength'.")

        if values['freq'] is None:
            values['freq'] = C_0 / values['wavelength']
        else:
            values['wavelength'] = C_0 / values['freq']

        return values

    def run(self, wavelength):
        if self.grid.num_cells.count(1) != 1:
            raise NotImplementedError("Actually, only 2D FDFD is supported!")

    # def _build_eps():

    # def _compute_source():

