from copy import deepcopy
import numpy as np
import scipy.sparse as sp

from .simulation import SimulationFdfd
from ..constants import EPS_0, MU_0, PI
from ..utils import fill_along_axis


import matplotlib.pyplot as plt


# TODO: Find a way to store the value of the source
def planewave_init(src, sim: SimulationFdfd):
    source_value = np.zeros(sim.grid.num_cells, dtype=complex)
    inj_idx = src.injection_axis
    indices = sim.grid.discretize_inds(src)

    # Planewave propagate along ALL the window by definition
    fill_along_axis(
        source_value,
        value=src.amplitude * np.exp(1j * src.phase),
        idx=indices[inj_idx], axis=inj_idx
    )

    if sim.tfsf is True:
        return qaaq_source(src, sim)
    else:
        return source_value


# dict_src_init = {'PlaneWave': planewave_init}
default_init = planewave_init
dict_src_init = default_init


def qaaq_source(src, sim: SimulationFdfd):
    mask = np.ones(sim.grid.num_cells, dtype=complex)

    if src.direction == '+':
        mask = fill_along_axis(
            array=mask, value=0.j,
            idx=range(sim.grid.discretize_inds(src)[src.injection_axis][0]),
            axis=src.injection_axis
        )
    else:
        mask = fill_along_axis(
            array=mask, value=0.j,
            idx=range(
                sim.grid.discretize_inds(src)[src.injection_axis][1],
                sim.grid.num_cells[src.injection_axis]
            ),
            axis=src.injection_axis
        )

    if sim.polarization == 'TE':
        coords_x = np.array(sim.grid.yee.E.z.x)
        coords_y = np.array(sim.grid.yee.E.z.y)
        mask = mask.squeeze()
    elif sim.polarization == 'TM':
        coords_x = np.array(sim.grid.yee.H.z.x)
        coords_y = np.array(sim.grid.yee.H.z.y)
        mask = mask.squeeze()
    else:
        raise NotImplementedError("Actually, only 2D is supported.")

    coords_x = coords_x - src.center[0]
    coords_y = coords_y - src.center[1]

    if src.phase == 0:
        fmag = src.amplitude
    else:
        fmag = src.amplitude * np.exp(1.j * src.phase)

    k_x, k_y, _ = _calc_k_tfsf(src, sim)

    xx, yy = np.meshgrid(coords_x, coords_y, indexing='ij')
    f = fmag * np.exp(1.j * k_x * xx) * np.exp(1.j * k_y * yy)
    return _solve_QAAQ(sim, src, mask, f)


def _solve_QAAQ(sim, src, Q, f):
    Fdfd_obj = deepcopy(sim.handler.FdfdSim)
    eps = next(iter(sim.intersecting_media(src, sim.structures))).permittivity
    Fdfd_obj.eps = np.ones_like(Fdfd_obj.eps) * eps
    sp_Q = sp.diags(Q.flatten(),
                    offsets=0,
                    shape=[Fdfd_obj.N] * 2,
                    dtype=complex)

    matQAAQ = sp_Q.dot(Fdfd_obj.A) - Fdfd_obj.A.dot(sp_Q)
    # Factor j / omega to keep the normalization
    return -matQAAQ.dot(f.flatten()) * 1.j / (2 * PI * sim.freq0)


# To make the TFSF working, it is necessary to use conj(bloch_conditions)
# + to take a vector flipped by -1
# No idea why actually...
def _calc_k_tfsf(src, sim):
    # 'intersecting_media' returns a set -> We use an iterator to access the value
    eps = next(iter(sim.intersecting_media(src, sim.structures))).permittivity

    kmag = 2 * PI * np.real(src.freq0 * np.sqrt(eps * EPS_0 * MU_0))
    # The standard units in Tidy3d are the micrometers
    # -> Need to modify the value of kmag to be right with others coords
    kmag *= 1e-6

    angle_theta = src.angle_theta
    angle_phi = src.angle_phi

    if src.direction == "-":
        angle_theta += PI

    if angle_theta == 0:
        k_local = [
            0,
            0,
            -kmag,
        ]
    else:
        k_local = [
            -kmag * np.sin(angle_theta) * np.cos(angle_phi),
            -kmag * np.sin(angle_theta) * np.sin(angle_phi),
            -kmag * np.cos(angle_theta),
        ]
    k_global = src.unpop_axis(
        k_local[2], (k_local[0], k_local[1]), axis=src.injection_axis
    )

    return k_global
