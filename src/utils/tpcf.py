from halotools.mock_observables import tpcf
import numpy as np


def compute_tpcf(positions: np.ndarray,
		boxsize:float = 302.6 ):
	'''
	Computes the real space two point correlation function using halotools

	Args:
		postions: 3D array with the cartesian coordiantes of the tracers.
		boxsize: box size of the simulation in the same units as positions. 

	'''


	r = np.geomspace(0.3, 30., 50)
	r_c = 0.5 * (r[1:] + r[:-1])


	real_tpcf = tpcf(positions, rbins = r,
				period = boxsize, estimator =  'Landy-Szalay' )

	return r_c, real_tpcf
