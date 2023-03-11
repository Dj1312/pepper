import numpy as np
import scipy.sparse as sp
from typing import NewType, Tuple
# from numpy.typing import NDArray

from .constants import EPS_0, ETA_0, FACTOR_M_FDFD, FACTOR_R_FDFD


# Define a "SSP" -> ScipySparse Array
SSPArray = NewType('SSPArray', sp.spmatrix)


# S_val calculations came form Jerry Shi and ceviche modules
# /!\ To modify since they are strangely centered -> an error is included"


def _calc_S_matrices(
        dl: Tuple[float, float], shape: Tuple[int, int],
        npml: Tuple[int, int],
        omega: float) -> Tuple[SSPArray, SSPArray, SSPArray, SSPArray]:
    dx, dy = dl
    Nx, Ny = shape
    Ntot = Nx * Ny

    Nx_pml, Ny_pml = npml
    Mat_Sxf = np.zeros(shape, dtype=complex)
    Mat_Sxb = np.zeros(shape, dtype=complex)
    Mat_Syf = np.zeros(shape, dtype=complex)
    Mat_Syb = np.zeros(shape, dtype=complex)

    for i in range(0, Ny):
        Mat_Sxf[:, i] = 1 / _calc_Svec_forward(dx, Nx, Nx_pml, omega)
        Mat_Sxb[:, i] = 1 / _calc_Svec_backward(dx, Nx, Nx_pml, omega)
    for i in range(0, Nx):
        Mat_Syf[i, :] = 1 / _calc_Svec_forward(dy, Ny, Ny_pml, omega)
        Mat_Syb[i, :] = 1 / _calc_Svec_backward(dy, Ny, Ny_pml, omega)

    _Sxf = sp.diags(Mat_Sxf.flatten(), 0, shape=(Ntot, Ntot), dtype=complex)
    _Sxb = sp.diags(Mat_Sxb.flatten(), 0, shape=(Ntot, Ntot), dtype=complex)
    _Syf = sp.diags(Mat_Syf.flatten(), 0, shape=(Ntot, Ntot), dtype=complex)
    _Syb = sp.diags(Mat_Syb.flatten(), 0, shape=(Ntot, Ntot), dtype=complex)

    return _Sxf, _Sxb, _Syf, _Syb


def _calc_Svec_forward(ds: float, Nshape: int,
                       npml: int, omega: float) -> SSPArray:
    dpml = ds * npml

    Svec = np.ones(Nshape, dtype=complex)
    for i in range(Nshape):
        if (i + 1) <= npml and npml != 0:
            # TODO: Modify the idx values (1) - Bottom part of PML
            Svec[i] = _calc_sw_val(npml - (i + 1.0) + 0.5,
                                   npml, dpml, omega)
        elif (i + 1) > Nshape - npml and npml != 0:
            # TODO: Modify the idx values (2) - Top part of PML
            Svec[i] = _calc_sw_val((i + 1.0) - (Nshape - npml) - 0.5,
                                   npml, dpml, omega)
    return Svec


def _calc_Svec_backward(ds: float, Nshape: int,
                        npml: int, omega: float) -> SSPArray:
    dpml = ds * npml

    Svec = np.ones(Nshape, dtype=complex)
    for i in range(Nshape):
        if (i + 1) <= npml and npml != 0:
            # TODO: Modify the idx values (3) - Bottom part of PML
            Svec[i] = _calc_sw_val(npml - (i + 1.0) + 1.0,
                                   npml, dpml, omega)
        elif (i + 1) > Nshape - npml and npml != 0:
            # TODO: Modify the idx values (4) - Top part of PML
            Svec[i] = _calc_sw_val((i + 1.0) - (Nshape - npml) - 1.0,
                                   npml, dpml, omega)
    return Svec


def _calc_sw_val(d: float, d_max: int, dpml: float, omega: float) -> SSPArray:
    # /!\ d_max differs from dpml /!\ (d = [cells], dpml = [meters])
    return 1 - 1.0j * _calc_sigma_val(d, d_max, dpml) / omega / EPS_0


def _calc_sigma_val(d: float, d_max: int, dpml: float) -> SSPArray:
    """Calculation of the Berenger's parameter for PML"""
    sig_max = -(FACTOR_M_FDFD + 1) * FACTOR_R_FDFD / (2 * ETA_0 * dpml)
    return sig_max * np.power(d / d_max, FACTOR_M_FDFD)
