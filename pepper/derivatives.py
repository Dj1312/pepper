import numpy as np
import scipy.sparse as sp
from typing import NewType, Tuple
# from numpy.typing import NDArray


# Define a "SSP" -> ScipySparse Array
SSPArray = NewType('SSPArray', sp.spmatrix)


def _calc_D_matrices(
        dl: Tuple[float, float],
        shape: Tuple[int, int]) -> Tuple[SSPArray, SSPArray, SSPArray, SSPArray]:
    """ All derivatives operators for both directions and schemes """
    # TODO: Explicit
    # Kind of cheat to obtain correct result :
    # + If the result is fully periodic, i.e. F(z) = F(z + Nz*dz), ok !
    # (Can be viewed as infinite on this particular direction)
    # + If not, the periodic result will vanish in PML F(z) = F(z + Nz*dz) = 0
    _Dxf = _calc_Dxf(dl, shape, periodic_phase=0.0)
    _Dxb = _calc_Dxb(dl, shape, periodic_phase=0.0)
    _Dyf = _calc_Dyf(dl, shape, periodic_phase=0.0)
    _Dyb = _calc_Dyb(dl, shape, periodic_phase=0.0)
    # _Dxf = _calc_Dxf(dl, shape, periodic_phase=None)
    # _Dxb = _calc_Dxb(dl, shape, periodic_phase=None)
    # _Dyf = _calc_Dyf(dl, shape, periodic_phase=None)
    # _Dyb = _calc_Dyb(dl, shape, periodic_phase=None)

    return _Dxf, _Dxb, _Dyf, _Dyb


def _calc_Dxf(dl: Tuple[float, float], shape: Tuple[int, int],
              periodic_phase: float = None) -> SSPArray:
    """ Derivative operator in x-direction with forward difference scheme """
    dx, _ = dl
    Nx, Ny = shape

    # If there is need of periodicity, add the correction into matrix
    if periodic_phase is None:
        Dxf = sp.diags([-1, 1], [0, 1], shape=(Nx, Nx))
    else:
        cor_per = np.exp(1.0j * periodic_phase)
        Dxf = sp.diags([-1, 1, cor_per], [0, 1, -Nx + 1], shape=(Nx, Nx))

    # Numpy is row major
    #   -> Two contiguous x-elements will be spaced by Ny-elements
    #      from the point of view of the flattened field
    Dxf = sp.kron(Dxf, sp.eye(Ny))
    Dxf = 1 / dx * Dxf

    return Dxf


def _calc_Dxb(dl: Tuple[float, float], shape: Tuple[int, int],
              periodic_phase: float = None) -> SSPArray:
    """ Derivative operator in x-direction with backward difference scheme """
    dx, _ = dl
    Nx, Ny = shape

    # If there is need of periodicity, add the correction into matrix
    if periodic_phase is None:
        Dxb = sp.diags([1, -1], [0, -1], shape=(Nx, Nx))
    else:
        cor_per = -np.exp(-1.0j * periodic_phase)
        Dxb = sp.diags([1, -1, cor_per], [0, -1, Nx - 1], shape=(Nx, Nx))

    # Numpy is row major
    #   -> Two contiguous x-elements will be spaced by Ny-elements
    #      from the point of view of the flattened field
    Dxb = sp.kron(Dxb, sp.eye(Ny))
    Dxb = 1 / dx * Dxb

    return Dxb


def _calc_Dyf(dl: Tuple[float, float], shape: Tuple[int, int],
              periodic_phase: float = None) -> SSPArray:
    """ Derivative operator in y-direction with forward difference scheme """
    _, dy = dl
    Nx, Ny = shape

    # If there is need of periodicity, add the correction into matrix
    if periodic_phase is None:
        Dyf = sp.diags([-1, 1], [0, 1], shape=(Ny, Ny))
    else:
        cor_per = np.exp(1.0j * periodic_phase)
        Dyf = sp.diags([-1, 1, cor_per], [0, 1, -Ny + 1], shape=(Ny, Ny))

    # Numpy is row major
    #   -> Two contiguous y-elements will be spaced by 1-element
    #      from the point of view of the flattened field
    Dyf = sp.kron(sp.eye(Nx), Dyf)
    Dyf = 1 / dy * Dyf

    return Dyf


def _calc_Dyb(dl: Tuple[float, float], shape: Tuple[int, int],
              periodic_phase: float = None) -> SSPArray:
    """ Derivative operator in y-direction with forward difference scheme """
    _, dy = dl
    Nx, Ny = shape

    # If there is need of periodicity, add the correction into matrix
    if periodic_phase is None:
        Dyb = sp.diags([1, -1], [0, -1], shape=(Ny, Ny))
    else:
        cor_per = -np.exp(-1.0j * periodic_phase)
        Dyb = sp.diags([1, -1, cor_per], [0, -1, Ny - 1], shape=(Ny, Ny))

    # Numpy is row major
    #   -> Two contiguous y-elements will be spaced by 1-element
    #      from the point of view of the flattened field
    Dyb = sp.kron(sp.eye(Nx), Dyb)
    Dyb = 1 / dy * Dyb

    return Dyb
