from enum import Enum
from functools import cached_property
from math import prod
from .constants import C_0, EPS_0, MU_0, PI
from .derivatives import _calc_D_matrices
from .pml import _calc_S_matrices
import scipy.sparse as sp
from tidy3d import Simulation as Tidy3dSim
from pyMKL import pardisoSolver


class SimulationType(Enum):
    FDTD = "fdtd"
    FDFD = "fdfd"


# class Simulation(Tidy3dSim):

#     def run(type: SimulationType, *args, **kwargs):
#         if type == SimulationType.FDTD:
#             raise NotImplementedError
#         elif type == SimulationType.FDFD:
#             print("Lessss go")
#         else:
#             raise ValueError

# Simple FDFD simu (no interest in mu right now)
# TODO: Refactor with a SimulationFdfd class (dataclass)
# TODO: Add a normalization system to formate unis and make EPS0 correct
# TODO: Allow other solver than pyMKL to be used
class SimulationFdfd_TE:
    def __init__(self, eps, dl, npml, omega=None, wavelength=None):
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

    @cached_property
    def D_ops(self):
        Dxf, Dxb, Dyf, Dyb = _calc_D_matrices([self.dl] * 2, self.shape)
        Sxf, Sxb, Syf, Syb = _calc_S_matrices([self.dl] * 2, self.shape,
                                              self.npml, self.omega)
        return {'SDxf': Sxf.dot(Dxf),
                'SDxb': Sxb.dot(Dxb),
                'SDyf': Syf.dot(Dyf),
                'SDyb': Syb.dot(Dyb)}

    @cached_property
    def sp_eps_zz(self):
        return sp.diags(self.eps.flatten(),
                        offsets=0,
                        shape=(self.N, self.N),
                        dtype=complex)

    @cached_property
    def A(self):
        # D = (1 / MU_0 * self.D_ops.get('SDxf').dot(self.D_ops.get('SDxb'))
        #      + 1 / MU_0 * self.D_ops.get('SDyf').dot(self.D_ops.get('SDyb')))
        # G = self.omega ** 2 * EPS_0 * self.sp_eps_zz
        # return D + G
        D = (- 1 / MU_0 * self.D_ops["SDxf"] @ self.D_ops["SDxb"]
             - 1 / MU_0 * self.D_ops["SDyf"] @ self.D_ops["SDyb"])
        G = - self.omega ** 2 * EPS_0 * self.sp_eps_zz
        return D + G


    @cached_property
    def b(self):
        return 1.0j * self.omega * self.source.flatten()

    def solve(self, source):
        self.source = source

        pSolve = pardisoSolver(self.A.tocsr(), mtype=13)
        pSolve.factor()
        x = pSolve.solve(self.b)
        pSolve.clear()
        self.EzF = x
        # self.EzF = spsolve(self.A, self.b)

        return self.EzF.reshape(self.shape)
