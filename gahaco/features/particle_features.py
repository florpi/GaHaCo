import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, newton
import sys 
sys.path.insert(0,"/cosma/home/dp004/dc-cues1/arepo_hdf5_library")
import read_hdf5


def nfw(r: np.array,
		rho_s: float,
		c: float):
	'''
	Returns the logarithm of the density at a given distance from the halo centre,
    based in an NFW profile.
	Args:
		r: distance to halo centre.
		rho_s: density paramter.
		c: concentration.
	Returns:
		log(density)

	'''
	return np.log10(rho_s) - np.log10(r*c) - (2*np.log10(1+(r*c)))


class ParticleSnapshot:
	'''
	Class to operate on TNG's particle data
	'''

	def __init__(self, h5_dir="/cosma7/data/TNG/TNG300-1-Dark/",
			snapshot_number=99):

		self.snapshot = read_hdf5.snapshot(snapshot_number, h5_dir, 
							check_total_particle_number=True)
		self.snapshot.group_catalog(['Group_M_Crit200',
						   'Group_R_Crit200',
						   'GroupLenType',
						   'GroupFirstSub',
						   'GroupPos',
						   'SubhaloVmax',
						   ])

		self.snapshot.read(['Coordinates'],parttype=[1])

		self.N_particles = self.snapshot.cat['GroupLenType'][:,1].astype(np.int64)
		self.firstsub = (self.snapshot.cat['GroupFirstSub']).astype(np.int64)
		self.cum_N_particles= np.cumsum(self.N_particles) - self.N_particles
		self.coordinates = self.snapshot.data['Coordinates']['dm'][:]

		self.h = self.snapshot.header.hubble
		# Get only resolved halos
		self.halo_mass_thresh = 1.0e11 
		self.halo_mass_cut = (
				self.snapshot.cat["Group_M_Crit200"][:]*self.snapshot.header.hubble>self.halo_mass_thresh
		)
		# Save IDs of haloes
		self.ID_DMO = np.arange(0, len(self.N_particles))
		self.ID_DMO = self.ID_DMO[self.halo_mass_cut]
		self.N_halos = len(self.ID_DMO)
		self.cum_N_particles = self.cum_N_particles[self.halo_mass_cut]
		self.halo_pos = self.snapshot.cat['GroupPos'][self.halo_mass_cut] 
		self.r200c = self.snapshot.cat['Group_R_Crit200'][self.halo_mass_cut]
		self.m200c = self.snapshot.cat['Group_M_Crit200'][self.halo_mass_cut]
		self.N_particles = self.N_particles[self.halo_mass_cut]
		self.firstsub = self.firstsub[self.halo_mass_cut]
		self.vmax = self.snapshot.cat['SubhaloVmax'][self.firstsub]

	def prada(self):
		'''
		Computes the concentration as in Prada et al. (2012) , instead of NFW fitting.
		Returns:
				Halo concentration

		'''
		scale_factor = 1./(1. + self.snapshot.header.redshift)
		r200c_physical = self.r200c*scale_factor/1000. # units Mpc

		v200 = ((self.snapshot.const.G * self.m200c)/r200c_physical*\
				self.snapshot.const.Mpc**2/1000.**2)**0.5  #units km/s

		def y(x, vmax, v200):
			func = np.log(1+x) - (x/(1+x))
			return ((0.216*x)/func)**0.5 - (vmax/v200)

					
		concentration = np.zeros((len(self.vmax)))
		for halo in range(self.N_halos):
			if v200[halo] > self.vmax[halo]:
				concentration[halo] = -9999.
			else:
				try:
					concentration[halo] = newton(y, x0=5., args=(self.vmax[halo], v200[halo]))
				except:
					concentration[halo] = -9999.

		return concentration

	def density_profile(self, halo_idx, nbins = 20):
		''' 
		Get density profile of halo with id halo_idx
		Args:
			halo_idx: halo id
			nbins: number of bins in the radial direction
		Returns:
			bin_radii; radial bin centers.
			bin_densities: density in randial bins
		'''

		min_rad = 0.05
		max_rad = 1
		bins = np.logspace(np.log10(min_rad),np.log10(max_rad),nbins+1,base=10.0)
		bin_radii = 0.5*(bins[1:]+bins[:-1])

		bin_volumes = 4./3.*np.pi*(bins[1:]**3 - bins[:-1]**3)

		coordinates_in_halo = self.coordinates[self.cum_N_particles[halo_idx]:self.cum_N_particles[halo_idx]+self.N_particles[halo_idx]]

		r_particle_halo = np.linalg.norm((coordinates_in_halo - self.halo_pos[halo_idx]),
			   axis = 1)/self.r200c[halo_idx]

		number_particles = np.histogram(r_particle_halo, bins=bins)[0]
		bin_masses = number_particles*self.Mpart
		bin_densities = bin_masses/bin_volumes

		return bin_radii, bin_densities

	def fit_nfw(self):
		'''
		Fit NFW profile to mesured density profiles in the simulation.
		Procedure defined in http://arxiv.org/abs/1104.5130.
		'''
		self.Mpart = self.snapshot.header.massarr[1] * 1e10 / self.h   # particle mass, Msun
		rho_crit = self.snapshot.const.rho_crit   # Msun Mpc^-3

		self.concentration = np.zeros((self.N_halos))
		self.rho_s = np.zeros((self.N_halos))
		self.chisq = np.zeros((self.N_halos))

		for halo in range(self.N_halos):
			bin_radii, bin_densities = self.density_profile(halo)
			fit_densities = bin_densities[bin_densities > 0.]
			fit_radii = bin_radii[bin_densities > 0.]

			if (len(fit_densities) > 2) & (self.N_particles[halo] >= 5000):
				try:
					popt, pcov = curve_fit(nfw, fit_radii, np.log10(fit_densities), p0=(8000*rho_crit, 5.))	
				except:
					popt = (-9999, -9999)

				self.rho_s[halo] = popt[0]
				self.concentration[halo] = popt[1]
				fit = nfw(fit_radii,*popt)
				self.chisq[halo] = 1/len(bin_radii)*np.sum((np.log10(fit_densities)-fit)**2) 

			else:
				self.rho_s[halo] = -9999
				self.concentration[halo] = -9999
				self.chisq[halo] = -9999



if __name__ == "__main__":

	snap = ParticleSnapshot()

	prada_concentration = snap.prada()
	snap.fit_nfw()
	# Save features to file
	
	features2save = np.vstack([snap.ID_DMO, prada_concentration,
		snap.concentration, snap.rho_s, snap.chisq, snap.m200c]).T

	print(features2save.shape)

	df = pd.DataFrame(
        data = features2save,
        columns = [
            'ID_DMO', 'concentration_prada', 'concentration_nfw', 'rho_s', 'chisq_nfw', 'm200c'
        ]
    )
	output_dir = '/cosma6/data/dp004/dc-cues1/tng_dataframes/'


	df.to_hdf(output_dir + 'halo_profiles.hdf5', key = 'df', mode = 'w')
	


	



