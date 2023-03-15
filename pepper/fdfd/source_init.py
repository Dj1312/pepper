import numpy as np


# TODO: Find a way to store the value of the source
def planewave_init(self, sim):
    source_value = np.zeros(sim.grid.num_cells, dtype=complex)
    inj_idx = self.injection_axis
    indices = sim.grid.discretize_inds(self)

    # Planewave propagate along ALL the window by definition

    # TODO: Use the function move or reshape to put the axis to not modify in last position
    # if inj_idx == 0:
    #     source_value[indices[0],:,:] = self.amplitude * np.exp(1j * self.phase)
    # elif inj_idx == 1:
    #     source_value[:,indices[1],:] = self.amplitude * np.exp(1j * self.phase)
    # elif inj_idx == 2:
    #     source_value[:,:,indices[2]] = self.amplitude * np.exp(1j * self.phase)
    source_value = np.moveaxis(source_value, source=inj_idx, destination=-1)
    source_value[:,:,indices[inj_idx]] = self.amplitude * np.exp(1j * self.phase)
    source_value = np.moveaxis(source_value, source=-1, destination=inj_idx)

    return source_value


# dict_src_init = {'PlaneWave': planewave_init}
default_init = planewave_init
dict_src_init = default_init
