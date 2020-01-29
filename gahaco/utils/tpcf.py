from halotools.mock_observables import tpcf
import numpy as np
from nbodykit.lab import *
from nbodykit.source.catalog.array import ArrayCatalog


def compute_tpcf(positions: np.ndarray, boxsize: float = 100.):
    """
    Computes the real space two point correlation function using halotools

    Args:
        postions: 3D array with the cartesian coordiantes of the tracers.
        boxsize: box size of the simulation in the same units as positions.

    """

    if boxsize < 150:
        r = np.geomspace(0.3, 10.0, 7)
    else:
        r = np.geomspace(0.3, 30.0, 20)
    r_c = 0.5 * (r[1:] + r[:-1])

    real_tpcf = tpcf(positions, rbins=r, period=boxsize, estimator="Landy-Szalay")

    return r_c, real_tpcf

def compute_power_spectrum(positions: np.ndarray, boxsize: float = 100.):
    """
    Computes the real space power spectrum using nbodykit 

    Args:
        postions: 3D array with the cartesian coordiantes of the tracers.
        boxsize: box size of the simulation in the same units as positions.

    """

    cat = ArrayCatalog({'pos':positions}, BoxSize = boxsize)
    mesh = cat.to_mesh(window='tsc', Nmesh=512, compensated=True, position='pos')

    r = FFTPower(mesh, mode='1d', dk=0.05, kmin=0.1)
    Pk = r.power

    return Pk

