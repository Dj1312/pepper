import numpy as np

from .simulation import SimulationFdfd


# TODO: Find a way to store the value of the source
def planewave_init(src, sim: SimulationFdfd):
    source_value = np.zeros(sim.grid.num_cells, dtype=complex)
    inj_idx = src.injection_axis
    indices = sim.grid.discretize_inds(src)

    # Planewave propagate along ALL the window by definition
    source_value = np.moveaxis(source_value, source=inj_idx, destination=-1)
    source_value[:,:,indices[inj_idx]] = src.amplitude * np.exp(1j * src.phase)
    source_value = np.moveaxis(source_value, source=-1, destination=inj_idx)

    if sim.tfsf is True:
        return qaaq_source(source_value, sim)
    else:
        return source_value


# dict_src_init = {'PlaneWave': planewave_init}
default_init = planewave_init
dict_src_init = default_init


def qaaq_source(src, src_value, sim: SimulationFdfd):
    mask = np.zeros_like(src_value)

    if sim.polarization == 'TE':
        coords_x = sim.grid.yee.E.z.x
        coords_y = sim.grid.yee.E.z.y
    elif sim.polarization == 'TM':
        coords_x = sim.grid.yee.H.z.x
        coords_y = sim.grid.yee.H.z.y
    else:
        raise NotImplementedError("Actually, only 2D is supported.")

    #TODO: What if the Medium has a 3 length ?
    #TODO: Finish the source
    # 'intersecting_media' returns a set -> We use an iterator to access the value
    eps = next(iter(sim.intersecting_media(src, sim.structures))).permittivity

    f = np.exp(1.j * k_x * coords_x) * np.exp(1.j * k_y * coords_y)

