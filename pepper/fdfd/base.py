from abc import ABC
from enum import Enum
from functools import cached_property
from math import prod

import numpy as np
import scipy.sparse as sp

from ..constants import C_0, EPS_0, MU_0, PI
from ..derivatives import _calc_D_matrices
from ..pml import _calc_S_matrices
from ..solver import linear_solve


class SimulationType(Enum):
    FDTD = "fdtd"
    FDFD = "fdfd"


# Simple FDFD simu (no interest in mu right now)
# TODO: Add a normalization system to formate unis and make EPS0 correct
# Transform to a dataclass ?
class BaseSimulationFdfd(ABC):
    def __init__(self, eps, dl, npml, omega=None, wavelength=None,
                 bloch_conditions=[None, None]):
        if omega is None and wavelength is None:
            raise ValueError("At least one of 'omega' or 'wavelength' must be provided.")
        elif omega is not None and wavelength is not None:
            raise ValueError("Only one of 'omega' or 'wavelength' should be provided.")
        elif omega is not None:
            self.omega = omega
            self.wavelength = 2 * PI * C_0 / omega
        else:
            self.wavelength = wavelength
            self.omega = 2 * PI * C_0 / wavelength

        self.eps = eps
        self.dl = dl
        self.npml = npml

        self.shape = eps.shape
        self.Nx, self.Ny = eps.shape
        self.N = prod(eps.shape)

        self.bloch_conditions = bloch_conditions

    @cached_property
    def D_ops(self):
        # We need to keep the value of D_ops to calculate the fields Fx, Fy
        Dxf, Dxb, Dyf, Dyb = _calc_D_matrices(self.dl, self.shape,
                                              self.bloch_conditions)
        Sxf, Sxb, Syf, Syb = _calc_S_matrices(self.dl, self.shape,
                                              self.npml, self.omega)
        return {
            'SDxf': Sxf.dot(Dxf),
            'SDxb': Sxb.dot(Dxb),
            'SDyf': Syf.dot(Dyf),
            'SDyb': Syb.dot(Dyb),
            'Dxf': Dxf,
            'Dxb': Dxb,
            'Dyf': Dyf,
            'Dyb': Dyb,
        }

    @cached_property
    def b(self):
        return -1.0j * self.omega * self.source.flatten()

    # TODO: Allow other solver than pyMKL to be used
    def solve(self, source=None):
        if source is None:
            source = self.b
        else:
            self.source = source
        x = linear_solve(self.A.tocsr(), self.b)
        Field_zF = x

        return Field_zF.reshape(self.shape)


class SimulationFdfd_TE(BaseSimulationFdfd):
    @cached_property
    def sp_eps_zz(self):
        return sp.diags(self.eps.flatten(),
                        offsets=0,
                        shape=(self.N, self.N),
                        dtype=complex)

    @cached_property
    def A(self):
        D = (1 / MU_0 * self.D_ops.get('SDxf').dot(self.D_ops.get('SDxb'))
             + 1 / MU_0 * self.D_ops.get('SDyf').dot(self.D_ops.get('SDyb')))
        G = self.omega ** 2 * EPS_0 * self.sp_eps_zz
        return D + G

    def solve(self, source):
        self.Ez = super().solve(source)
        return self.Ez


class SimulationFdfd_TM(BaseSimulationFdfd):
    @cached_property
    def sp_inv_eps_xx(self):
        eps_xx = 1 / 2 * (self.eps + np.roll(self.eps, axis=1, shift=1))
        return sp.diags(np.reciprocal(eps_xx.flatten()),
                        offsets=0,
                        shape=(self.N, self.N),
                        dtype=complex)

    @cached_property
    def sp_inv_eps_yy(self):
        eps_yy = 1 / 2 * (self.eps + np.roll(self.eps, axis=0, shift=1))
        return sp.diags(np.reciprocal(eps_yy.flatten()),
                        offsets=0,
                        shape=(self.N, self.N),
                        dtype=complex)

    @cached_property
    def A(self):
        D = (1 / EPS_0 * self.D_ops.get('SDxf').dot(self.sp_inv_eps_yy).dot(self.D_ops.get('SDxb'))
             + 1 / EPS_0 * self.D_ops.get('SDyf').dot(self.sp_inv_eps_xx).dot(self.D_ops.get('SDyb')))
        G = self.omega ** 2 * MU_0 * sp.eye(m=self.N, dtype=complex)
        return D + G

    def solve(self, source):
        self.Hz = super().solve(source)
        return self.Hz
