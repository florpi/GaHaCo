import os, sys
import numpy as np
import h5py
import pandas as pd
from astropy import units
from astropy.constants import G
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

    def __init__(
        self,
        overdensity_radius=2500,
        h5_dir="/cosma7/data/TNG/TNG300-1-Dark/",
        snapshot_number=99
        ):
        """
        Load data of particles that belong to resolved halos
        Args:
            overdensity_radius: int
                Out to which radius halo properties are measured.
                R200 - default SubFind radii
                R2500 - where galaxy formation takes place
            h5_dir: str
                path to particle data
            snapshot_number: int
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
        
        # dm particle mass
        self.Mpart = (
            self.snapshot.header.massarr[1] * 1e10 / self.h
        )  #[Msun] particle mass
       
        # if halo properties aren't to be found at r200c
        self.overdensity = overdensity
        if self.overdensity == 2500:
            self.find_overdensity_radius()
       

    def find_overdensity_radius(self):
        """
        """
        nfw_file_path = "/cosma7/data/dp004/dc-cues1/tng_dataframes/halo_nfw_profiles.hdf5"
        
        if not os.path.isfile(nfw_file_path):
            raise ImportError("File with NFW-profiles can't be opened")

        with h5py.File(nfw_file_path, 'r') as hf:
            nfw = {}
            for k in hf.keys():
                nfw[k] = hf.get(k).value


        self.overdensity_radius = np.ones((self.N_halos)) * -9999
        for halo_idx in range(self.N_halos):
            nfw_values = nfw['values'][halo_idx, ::-1]
            nfw_radii = nfw['radii'][halo_idx, ::-1]

            # only procede if NFW-profile was found
            if (-9999 in np.unique(nfw_values)) and (len(np.unique(nfw_values)) == 1):
                continue

            r200c_physical = s.cat["Group_R_Crit200"][halo_idx]*1e-3  #[Mpc]
            nfw_density = 10**nfw_values/r200c_physical**3  #[Msun/Mpc^3]
            delta_rho = nfw_density/rho_crit

            spl = BSpline(
                delta_rho,
                r200c_physical*nfw_radii,
                k=5
            )
            self.overdensity_radius[halo_idx] = spl(2500)  #[Mpc]


    def concentration(self, method='prada'):
        """
        Args:
            method: string
                ['prada', 'faltenbacher', 'nfw']

		Returns:
				Halo concentration

		"""
        if self.overdensity_radius is  2500:
            raise Exception(
                "The Prada concentration can currently only be found for R200"
            )

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


    def get_all_profiles(self, nbins=20):
        """
		Get density profiles of all halos

		Returns:
			bin_radii; radial bin centers.
			bin_densities: density in randial bins
		"""
        import gahaco.features.utils.profile as prof
        
        nbins = 20
        self.profiles_value = np.zeros((self.N_halos, nbins))
        self.profiles_radii = np.zeros((self.N_halos, nbins))

        for halo_idx in range(self.N_halos):
            bin_radii, bin_densities = self.get_one_profile(halo_idx, 'density', nbins)
            self.profiles_radii[halo_idx, :] = bin_radii
            self.profiles_value[halo_idx, :] = bin_densities


    def get_one_profile(self, halo_idx, quantity, nbins=20):
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
            bin_radii, bin_densities = prof.from_particle_data(
                rel_par_pos, rel_par_vel, 0, quantity, nbins,
            )
        else:
            bin_radii, bin_densities = prof.from_particle_data(
                rel_par_pos, 0, self.Mpart, quantity, nbins,
            )

        return bin_radii, bin_densities


    def fit_nfw(self):
        """
		Fit NFW profile to mesured density profiles in the simulation.
		Procedure defined in Prada et al. (2012)
        (DOI: 10.1111/j.1365-2966.2012.21007.x ; arxiv: 1104.5130)
		"""
        if self.overdensity_radius is  2500:
            raise Exception(
                "The NFW should only be measured out to R200" 
            )
        
        self.Mpart = (
            self.snapshot.header.massarr[1] * 1e10 / self.h
        )  #[Msun] particle mass
        rho_crit = self.snapshot.const.rho_crit  #[Msun/Mpc^3]

        self.concentration = np.zeros((self.N_halos))
        self.rho_s = np.zeros((self.N_halos))
        self.chisq = np.zeros((self.N_halos))
        nbins = 20
        self.nfw_profiles_value = np.zeros((self.N_halos, nbins))
        self.nfw_profiles_radii = np.zeros((self.N_halos, nbins))

        for halo_idx in range(self.N_halos):
            bin_radii, bin_densities = self.get_one_profile(halo_idx, 'density', nbins)

            # use only radii-bins to fit nfw, where a measurement exists
            fit_densities = bin_densities[bin_densities > 0.0]
            fit_radii = bin_radii[bin_densities > 0.0]

            if (len(fit_densities) > 2) & (self.N_particles[halo_idx] >= 5000):
                try:
                    popt, pcov = curve_fit(
                        nfw,
                        fit_radii,
                        np.log10(fit_densities),
                        p0=(8000 * rho_crit, 5.0),
                    )
                except:
                    popt = (-9999, -9999)

                self.rho_s[halo_idx] = popt[0]
                self.concentration[halo_idx] = popt[1]

                # use all radii-bins to create nfw-profile
                fit = nfw(bin_radii, *popt)
                self.nfw_profiles_radii[halo_idx, :] = bin_radii
                self.nfw_profiles_value[halo_idx, :] = fit
                self.chisq[halo_idx] = (
                    1 / len(bin_radii) * np.sum((np.log10(fit_densities) - fit) ** 2)
                )

            else:
                self.rho_s[halo_idx] = -9999
                self.concentration[halo_idx] = -9999
                self.chisq[halo_idx] = -9999
                self.nfw_profiles_radii[halo_idx, :] = -9999
                self.nfw_profiles_value[halo_idx, :] = -9999

            
    def velocity_anisotropy(self, radius):
        """
        Get the velocity anisotropy parameter.
        (DOI: 10.1016/j.nuclphysbps.2009.07.010; arxiv: 0810.3676)
		"""
        self.vel_ani_param = np.zeros(self.N_halos)

        for halo in range(self.N_halos):

            if self.N_particles[halo] >= 1000:
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
            
                if hasattr(self, 'overdensity_radius'):
                    # select within overdensity_radius
                    obj_part_dist = np.sqrt(
                        obj_part_pos[:, 0]**2 + \
                        obj_part_pos[:, 1]**2 + \
                        obj_part_pos[:, 2]**2
                    )
                    indx = obj_part_dist < self.overdensity_radius[halo]
                    obj_part_pos = obj_part_pos[indx]
                    obj_part_vel = obj_part_vel[indx]
            
                    if np.max(obj_part_dist.shape) < 100:
                        # if resolution too low, skip
                        continue

                vel_norm = np.linalg.norm(obj_part_vel, axis=1)
                pos_norm = np.linalg.norm(obj_part_pos, axis=1)

                # vector norms, thus only positive
                vr = np.abs((obj_part_pos * obj_part_vel).sum(axis=1) / pos_norm)
                vt = np.sqrt(vel_norm**2 - vr**2)

                sigma_r = np.std(vr)
                sigma_t = np.std(vt)

                self.vel_ani_param[halo] = 1 - 0.5*(sigma_t**2 / sigma_r**2)

            else:
                self.vel_ani_param[halo] = -9999
    
    
    def deltarho_summary_stats(self, radius):
        """
        Get the velocity anisotropy parameter.
        (DOI: 10.1016/j.nuclphysbps.2009.07.010; arxiv: 0810.3676)
		"""
        self.vmax = np.zeros(self.N_halos)
        self.mass = np.zeros(self.N_halos)
        self.vrms = np.zeros(self.N_halos)

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
        
            if hasattr(self, 'overdensity_radius'):
                # select within overdensity_radius
                obj_part_dist = np.sqrt(
                    obj_part_pos[:, 0]**2 + \
                    obj_part_pos[:, 1]**2 + \
                    obj_part_pos[:, 2]**2
                )
                indx = obj_part_dist < self.overdensity_radius[halo]
                obj_part_pos = obj_part_pos[indx]
                obj_part_vel = obj_part_vel[indx]
        
                if len(obj_part_pos) < 100:
                    # if resolution too low, skip
                    self.vmax[halo] = -9999
                    self.vrms[halo] = -9999
                    self.mass[halo] = -9999
                    continue

            vel_norm = np.linalg.norm(obj_part_vel, axis=1)

            self.vmax[halo] = _get_vmax(obj_part_pos)
            self.vrms[halo] = np.std(vel_norm)/np.sqrt(3)  # TODO double check
            self.mass[halo] = len(obj_part_vel) * self.Mpart


if __name__ == "__main__":

    snap = ParticleSnapshot(overdensity_radius=2500)

    # calculate properties
    #snap.concentration('prada')
    snap.fit_nfw()
    #snap.velocity_anisotropy()
    #snap.get_profile()
    #snap.r2500_summary()

    #
    # Store scalar features in pandas dataframe
    #
    #features2save = np.vstack(
    #    [
    #        snap.ID_DMO,
    #        prada_concentration,
    #        snap.concentration,
    #        snap.rho_s,
    #        snap.chisq,
    #        snap.m200c,
    #    ]
    #).T
    #df = pd.DataFrame(
    #    data=features2save,
    #    columns=[
    #        "ID_DMO",
    #        "concentration_prada",
    #        "concentration_nfw",
    #        "rho_s",
    #        "chisq_nfw",
    #        "m200c",
    #    ],
    #)
    
    ## Save features to file
    output_dir = "/cosma7/data/dp004/dc-cues1/tng_dataframes/"
    #df.to_hdf(output_dir + "halo_particle_summary.hdf5", key="df", mode="w")


    #
    # Store vector features in h5py
    #
    hf = h5py.File(output_dir + "halo_nfw_profiles.hdf5", 'w')
    hf.create_dataset('ID_DMO', data=snap.ID_DMO)
    hf.create_dataset('radii', data=snap.nfw_profiles_radii)
    hf.create_dataset('values', data=snap.nfw_profiles_value)
    hf.close()
