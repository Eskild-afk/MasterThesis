import numpy as np

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def pos (x):
    return np.maximum(x,0)

def neg (x):
    return np.minimum(x,0)

