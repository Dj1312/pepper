import numpy as np


def fill_along_axis(array, value, idx, axis):
    new_array = np.moveaxis(array, source=axis, destination=-1)
    new_array[:,:,idx] = value
    new_array = np.moveaxis(new_array, source=-1, destination=axis)
    return new_array