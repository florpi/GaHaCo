import sys
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, newton

sys.path.insert(0, "/cosma/home/dp004/dc-cues1/arepo_hdf5_library")
import read_hdf5


def nfw(r: np.array, rho_s: float, c: float):
    """
	Returns the logarithm of the density at a given distance from the halo centre,
    based in an NFW profile.
	Args:
		r: distance to halo centre.
		rho_s: density paramter.
		c: concentration.
	Returns:
		log(density)

	"""
    return np.log10(rho_s) - np.log10(r * c) - (2 * np.log10(1 + (r * c)))


class ParticleSnapshot:
    """
	Class to operate on TNG's particle data
	"""

    def __init__(self, h5_dir="/cosma7/data/TNG/TNG300-1-Dark/", snapshot_number=99):
        """
        Load data of particles that belong to resolved halos
        """

        self.snapshot = read_hdf5.snapshot(
            snapshot_number, h5_dir, check_total_particle_number=True
        )
        
        # read simulation settings
        self.h = self.snapshot.header.hubble

        # read subfind & fof data
        self.snapshot.group_catalog(
            [
                "Group_M_Crit200",
                "Group_R_Crit200",
                "GroupLenType",
                "GroupFirstSub",
                "GroupPos",
                "GroupVel",
                "SubhaloVmax",
            ]
        )
        # associate particles to subfind & fof objects
		self.N_particles = self.snapshot.cat['GroupLenType'][:,1].astype(np.int64)
		self.firstsub = (self.snapshot.cat['GroupFirstSub']).astype(np.int64)
		self.cum_N_particles= np.cumsum(self.N_particles) - self.N_particles
		self.ID_DMO = np.arange(0, len(self.N_particles))
		# get only resolved halos
		self.halo_mass_thresh = 1.0e11 
		self.halo_mass_cut = (
				self.snapshot.cat["Group_M_Crit200"][:]*self.snapshot.header.hubble>self.halo_mass_thresh
		)
		# filter subfind & fof objects
		self.N_particles = self.N_particles[self.halo_mass_cut]
		self.firstsub = self.firstsub[self.halo_mass_cut]
		self.cum_N_particles = self.cum_N_particles[self.halo_mass_cut]
		self.ID_DMO = self.ID_DMO[self.halo_mass_cut]
        self.halo_pos = self.snapshot.cat['GroupPos'][self.halo_mass_cut] 
        self.halo_vel = self.snapshot.cat['GroupVel'][self.halo_mass_cut] 
		self.r200c = self.snapshot.cat['Group_R_Crit200'][self.halo_mass_cut]
		self.m200c = self.snapshot.cat['Group_M_Crit200'][self.halo_mass_cut]
		self.vmax = self.snapshot.cat['SubhaloVmax'][self.firstsub]
		self.N_halos = len(self.vmax)

        # read particle data
        self.snapshot.read(["Coordinates", "Velocities"], parttype=[1])
		self.coordinates = self.snapshot.data['Coordinates']['dm'][:]
		self.velocities = self.snapshot.data['Velocities']['dm'][:]


    def concentration(self, method='prada'):
        """
        Args:
            method: string
                ['prada', 'faltenbacher', 'nfw']

		Returns:
				Halo concentration

		"""
        scale_factor = 1.0 / (1.0 + self.snapshot.header.redshift)
        r200c_physical = self.r200c * scale_factor / 1000.0  # units Mpc

        v200 = (
            (self.snapshot.const.G * self.m200c)
            / r200c_physical
            * self.snapshot.const.Mpc ** 2
            / 1000.0 ** 2
        ) ** 0.5  # units km/s

        def y(x, vmax, v200):
            func = np.log(1 + x) - (x / (1 + x))
            return ((0.216 * x) / func) ** 0.5 - (vmax / v200)

        concentration = np.zeros((len(self.vmax)))
        for halo in range(self.N_halos):
            if v200[halo] > self.vmax[halo]:
                concentration[halo] = -9999.0
            else:
                try:
                    concentration[halo] = newton(
                        y, x0=5.0, args=(self.vmax[halo], v200[halo])
                    )
                except:
                    concentration[halo] = -9999.0

        return concentration

    def profile(self, halo_idx, quantity, nbins=20):
        """
		Get density profile of halo with id halo_idx
		Args:
			halo_idx: int
                The halo id
            quantity: str
                One of [count, mass, temperature, velocity dispersion,
                pairwise radial velocity dispersion,]
			nbins: number of bins in the radial direction

		Returns:
			bin_radii; radial bin centers.
			bin_densities: density in randial bins
		"""
        import gahaco.features.utils.profile as prof

        # particles that belong to halo halo_idx
        coordinates_in_halo = self.coordinates[
            self.cum_N_particles[halo_idx] : self.cum_N_particles[halo_idx]
            + self.N_particles[halo_idx]
        ]

        # particle distances w.r.t halo centre
        rel_par_pos = (
            np.linalg.norm((coordinates_in_halo - self.halo_pos[halo_idx]), axis=1)
            / self.r200c[halo_idx]
        )

        if 'velo' in quantity:
            # particle velocity w.r.t halo
            rel_par_vel = (
                np.linalg.norm((coordinates_in_halo - self.halo_pos[halo_idx]), axis=1)
                / self.r200c[halo_idx]
            )
        
        prof.from_particle_data(rel_par_pos, rel_par_vel, quantity)

        return bin_radii, bin_densities

    def fit_nfw(self):
        """
		Fit NFW profile to mesured density profiles in the simulation.
		Procedure defined in Prada et al. (2012)
        (DOI: 10.1111/j.1365-2966.2012.21007.x ; arxiv: 1104.5130)
		"""
        self.Mpart = (
            self.snapshot.header.massarr[1] * 1e10 / self.h
        )  # particle mass, Msun
        rho_crit = self.snapshot.const.rho_crit  # Msun Mpc^-3

        self.concentration = np.zeros((self.N_halos))
        self.rho_s = np.zeros((self.N_halos))
        self.chisq = np.zeros((self.N_halos))

        for halo in range(self.N_halos):
            bin_radii, bin_densities = self.density_profile(halo)
            fit_densities = bin_densities[bin_densities > 0.0]
            fit_radii = bin_radii[bin_densities > 0.0]

            if (len(fit_densities) > 2) & (self.N_particles[halo] >= 5000):
                try:
                    popt, pcov = curve_fit(
                        nfw,
                        fit_radii,
                        np.log10(fit_densities),
                        p0=(8000 * rho_crit, 5.0),
                    )
                except:
                    popt = (-9999, -9999)

                self.rho_s[halo] = popt[0]
                self.concentration[halo] = popt[1]
                fit = nfw(fit_radii, *popt)
                self.chisq[halo] = (
                    1 / len(bin_radii) * np.sum((np.log10(fit_densities) - fit) ** 2)
                )

            else:
                self.rho_s[halo] = -9999
                self.concentration[halo] = -9999
                self.chisq[halo] = -9999


    def principal_axis(self):
        """
		"""
        import gahaco.features.utils import shape
        
        for halo in range(self.N_halos):
            # Find all particles in this object
            coordinates_halo = self.coordinates[
            self.group_offset[halo] : self.group_offset[halo] + self.N_particles[halo], :
            ]

            # Particle positons relative to object centre
            rel_part_pos = (coordinates_halo - self.halo_pos[halo]) / self.r200c[halo]

            # radii in which to find principal axis
            logr=np.linspace(np.log10(rmin),np.log10(rmax),nbins)
            rin=10**(logr-binwidth/2.)
            rout=10**(logr+binwidth/2.)

            if useR200==True:
                pos/=rvir
            else:
                rin=rin*300
                rout=rout*300
                rvir=1.

            self.principal_axis = shape.principal_axis(
                rel_part_pos, rvir, rin, rout, False, False, 1e-2
            )
    
            
    def velocity_anisotropy(self):
        """
        Get the velocity anisotropy parameter.
        (DOI: 10.1016/j.nuclphysbps.2009.07.010; arxiv: 0810.3676)
		"""
        self.vel_ani_param = np.zeros(self.N_halos)

        for halo in range(self.N_halos):
            # Find all particles in this object
            obj_part_pos = self.coordinates[
                self.group_offset[halo] : self.group_offset[halo] + self.N_particles[halo], :
            ]
            obj_part_vel = self.velocities[
                self.group_offset[halo] : self.group_offset[halo] + self.N_particles[halo], :
            ]

            # Particle properties w.r.t objects
            obj_part_pos = (obj_part_pos - self.halo_pos[halo]) / self.r200c[halo]
            obj_part_vel -= self.halo_vel[halo]

            # find angle between position & velocity vector
            phi = np.arccos(
                np.dot(obj_part_vel, obj_part_pos) /
                (np.linalg.norm(obj_part_vel) * np.linalg.norm(obj_part_pos))
            )

            # radial velocity dispersion
            sigma_r = np.std(np.linalg.norm(v) * np.cos(phi))

            # trangential velocity dispersion
            sigma_t = np.std(np.linalg.norm(v) * np.sin(phi))

            # the velocity anisotropy parameter
            self.vel_ani_param[halo] = 1 - 0.5*(sigma_t**2/sigma_r**2)


if __name__ == "__main__":

    snap = ParticleSnapshot()

    # calculate properties
    snap.concentration('prada')
    snap.principal_axis()
    snap.fit_nfw()
    snap.velocity_anisotropy()

    # store data in pandas dataframe
    features2save = np.vstack(
        [
            snap.ID_DMO,
            prada_concentration,
            snap.concentration,
            snap.rho_s,
            snap.chisq,
            snap.m200c,
        ]
    ).T
    print(features2save.shape)
    df = pd.DataFrame(
        data=features2save,
        columns=[
            "ID_DMO",
            "concentration_prada",
            "concentration_nfw",
            "rho_s",
            "chisq_nfw",
            "m200c",
        ],
    )
    
    # Save features to file
    output_dir = "/cosma6/data/dp004/dc-cues1/tng_dataframes/"
    df.to_hdf(output_dir + "halo_profiles.hdf5", key="df", mode="w")
