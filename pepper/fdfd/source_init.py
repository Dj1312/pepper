from copy import deepcopy
import numpy as np
import scipy.sparse as sp

from .simulation import SimulationFdfd
from ..constants import EPS_0, MU_0, PI, MICROMETERS
from ..utils import fill_along_axis


# TODO: Find a way to store the value of the source
def planewave_init(src, sim: SimulationFdfd):
    if sim.tfsf is True:
        return qaaq_source(linesource(src, sim), src, sim)
    else:
        return linesource(src, sim)


def gaussianbeam_init(src, sim: SimulationFdfd):
    if sim.tfsf is True:
        return qaaq_source(gaussian_beam(src, sim), src, sim)
    else:
        return gaussian_beam(src, sim)


def gaussian_beam(src, sim: SimulationFdfd):
    if sim.polarization == 'TE':
        _, coords = src.pop_axis(sim.discretize(src).yee.E.z.to_list, axis=src.injection_axis)
    elif sim.polarization == 'TM':
        _, coords = src.pop_axis(sim.discretize(src).yee.H.z.to_list, axis=src.injection_axis)
    else:
        raise NotImplementedError("Actually, only 2D is supported.")
    _, center = src.pop_axis(src.center, axis=src.injection_axis)
    # Centering the coords
    coords = [coords[i] - center[i] for i in range(len(center))]

    ii, jj = np.meshgrid(*coords, indexing='ij')
    r2 = ii**2 + jj**2

    # GaussianBeam parameters
    w0 = src.waist_radius
    z = src.waist_distance

    z0 = PI * src.waist_radius**2 / sim.wavelength
    w = w0 * np.sqrt(1 + (z / z0)**2)

    beam = w0 / w * np.exp(-r2 / (w**2))

    # Avoid rounding issue
    if z != 0.0:
        eps = next(iter(sim.intersecting_media(src, sim.structures))).permittivity
        k = 2 * PI / sim.wavelength * np.sqrt(eps)
        gouy = np.arctan(z / z0)
        r_curvature = z * (1 + (z0 / z)**2)
        add_phase = np.exp(-1.j * (
            k * z * MICROMETERS + k * r2 / (2 * r_curvature) - gouy
        ))
        beam = beam * add_phase

    # Avoid rounding issue
    if src.phase == 0.0:
        fmag = src.amplitude
    else:
        fmag = src.amplitude * np.exp(1.j * src.phase)

    source_value = np.zeros(sim.grid.num_cells, dtype=complex)
    indices = [np.arange(*val) for val in sim.grid.discretize_inds(src)]
    source_value[np.ix_(*indices)] = fmag * beam

    return source_value


# Why conj necessary ? -> e^j(wt-kx)
def linesource(src, sim):
    linesource = np.zeros(sim.grid.num_cells, dtype=complex)
    indices = [np.arange(*val) for val in sim.grid.discretize_inds(src)]

    # Avoid rounding issue
    if src.phase == 0.0:
        fmag = src.amplitude
    else:
        fmag = src.amplitude * np.exp(1.j * src.phase)

    linesource[np.ix_(*indices)] = fmag

    return linesource


def qaaq_source(src_value, src, sim: SimulationFdfd):
    num_cell_axis, num_cells_src = src.pop_axis(sim.grid.num_cells, axis=src.injection_axis)
    _, coords_src = src.pop_axis(sim.discretize(src).yee.E.z.to_list, axis=src.injection_axis)
    _, idx_src = src.pop_axis(sim.grid.discretize_inds(src), axis=src.injection_axis)

    src_slice = np.zeros(num_cells_src, dtype=complex)
    idx_slice_3d = [np.arange(*val) for val in sim.grid.discretize_inds(src)]
    idx_slice_2d = [np.arange(*val) for val in idx_src]

    # Get the 3D value of the source
    src_3D_slice = src_value[np.ix_(*idx_slice_3d)]
    # Since the injection axis is 1, we can squeeze it out to back to 2D
    src_slice[np.ix_(*idx_slice_2d)] = np.squeeze(src_3D_slice, axis=src.injection_axis)

    # After having a slice, the source can be extended on the full window
    src_extended = np.moveaxis(
        np.stack([src_slice] * num_cell_axis),
        source=0, destination=src.injection_axis
    )

    # Add the phase accumulation term
    src_with_phase = src_extended * calc_phase(src, sim)

    return _calc_QAAQ(src, sim, _calc_mask(src, sim), src_with_phase)


def calc_phase(src, sim):
    # TODO: Do we need to use Yee grid ?
    # A good idea can be to induse the shift directly when doing the simulation
    # Shift -> exp(-1j * k_i * dl_i_yee_grid)
    if sim.polarization == 'TE':
        coords_x = np.array(sim.grid.yee.E.z.x)
        coords_y = np.array(sim.grid.yee.E.z.y)
        coords_z = np.array(sim.grid.yee.E.z.z)
    elif sim.polarization == 'TM':
        coords_x = np.array(sim.grid.yee.H.z.x)
        coords_y = np.array(sim.grid.yee.H.z.y)
        coords_z = np.array(sim.grid.yee.H.z.z)
    else:
        raise NotImplementedError("Actually, only 2D is supported.")

    # coords_x, coords_y, coords_z = sim.grid.boundaries.to_list
    coords_x = coords_x - src.center[0]
    coords_y = coords_y - src.center[1]
    coords_z = coords_z - src.center[2]

    # # Avoid rounding issue
    # if src.phase == 0.0:
    #     fmag = src.amplitude
    # else:
    #     fmag = src.amplitude * np.exp(1.j * src.phase)

    k_x, k_y, k_z = _calc_k(src, sim)

    xx, yy, zz = np.meshgrid(coords_x, coords_y, coords_z, indexing='ij')
    f_phase = (
        np.exp(-1.j * k_x * xx) * np.exp(-1.j * k_y * yy) * np.exp(-1.j * k_z * zz)
    )

    return f_phase


def _calc_mask(src, sim):
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
    return mask


# Why conj necessary ? -> e^j(wt-kx)
def _calc_QAAQ(src, sim, Q, f):
    Fdfd_obj = deepcopy(sim.handler.FdfdSim)
    eps = next(iter(sim.intersecting_media(src, sim.structures))).permittivity
    Fdfd_obj.eps = np.ones_like(Fdfd_obj.eps) * eps
    sp_Q = sp.diags(Q.flatten(),
                    offsets=0,
                    shape=[Fdfd_obj.N] * 2,
                    dtype=complex)

    matQAAQ = sp_Q.dot(Fdfd_obj.A) - Fdfd_obj.A.dot(sp_Q)
    # Factor -j / omega to keep the normalization
    norm_factor = -1.j / (2 * PI * sim.freq0)
    return matQAAQ.dot(f.flatten()) * norm_factor


def _calc_k(src, sim):
    # 'intersecting_media' returns a set -> We use an iterator to access the value
    eps = next(iter(sim.intersecting_media(src, sim.structures))).permittivity

    kmag = 2 * PI * np.real(src.freq0 * np.sqrt(eps * EPS_0 * MU_0))
    #  The standard units in Tidy3d are the micrometers
    # -> Need to modify the value of kmag to be right with others coords
    kmag *= 1e-6

    angle_theta = src.angle_theta
    angle_phi = src.angle_phi

    if src.direction == "-":
        angle_theta += PI

    if angle_theta == 0.0:
        k_local = [
            0,
            0,
            kmag,
        ]
    else:
        k_local = [
            kmag * np.sin(angle_theta) * np.cos(angle_phi),
            kmag * np.sin(angle_theta) * np.sin(angle_phi),
            kmag * np.cos(angle_theta),
        ]

    k_global = src.unpop_axis(
        k_local[2], (k_local[0], k_local[1]), axis=src.injection_axis
    )

    return k_global


dict_src_init = {
    'PlaneWave': planewave_init,
    'UniformCurrentSource': linesource,
    # 'PointDipole': point_dipole_init,
    'GaussianBeam': gaussianbeam_init,
}
