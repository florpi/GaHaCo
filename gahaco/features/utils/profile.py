import sys
import numpy as np
import pynbody

sys.path.insert(0, "/cosma/home/dp004/dc-cues1/arepo_hdf5_library")
import read_hdf5


def from_particle_data(pos, vel, Mpart, quantity, nbins):
    """
    Get profile of halo
    Args:
        quantity: str
            One of [count, mass, temperature, velocity dispersion,
            pairwise radial velocity dispersion,]
    """

    # create radii bins
    min_rad = 0.05
    max_rad = 1
    nbins = 20
    bins = np.logspace(np.log10(min_rad), np.log10(max_rad), nbins + 1, base=10.0)
    bin_radii = 0.5 * (bins[1:] + bins[:-1])

    # volumne enclosed by each radii bin, normalized by r200
    bin_volumes = 4.0 / 3.0 * np.pi * (bins[1:] ** 3 - bins[:-1] ** 3)

    # profile
    if quantity == 'density':
        # only for dm
        number_particles = np.histogram(pos, bins=bins)[0]
        bin_masses = number_particles * Mpart  #[Msun]
        
        bin_value = bin_masses / bin_volumes
    
    elif quantity == 'velocity anisotropy':
        pass
    
    elif quantity == 'velocity dispersion':
        pass

    elif quantity == 'pairwise radial velocity dispersion':
        pass
        # TODO
        #from halotools.mock_observables import radial_pvd_vs_r
        #sigma_12 = radial_pvd_vs_r(
        #    rel_par_pos, rel_par_vel, rbins_absolute=rbin_edges, period=Lbox
        #)

    return bin_radii, bin_value


def _velocity_anisotropy(pos, vel, bins):

    # sort particles into radii bins
    indx = np.digitize(np.linalg.norm(pos), bins, right=True)

    beta = np.zeros(len(bins) - 1)
    for b in range(1, len(bins)):
        # select particles in shell
        pos_shell = pos[indx == b]
        vel_shell = vel[indx == b]

        # find angle between position & velocity vector
        phi = np.arccos(
            np.dot(vel_shell, pos_shell) / (np.linalg.norm(pos_shell) * np.linalg.norm(vel_shell))
        )

        # radial velocity dispersion
        sigma_r = np.std(np.linalg.norm(v) * np.cos(phi))

        # trangential velocity dispersion
        sigma_t = np.std(np.linalg.norm(v) * np.sin(phi))

        # the velocity anisotropy parameter
        beta[b] = 1 - 0.5*(sigma_t**2/sigma_r**2)
    
    return beta


def _velocity_dispersion(pos, vel, bins):
    """For the case that principal axis are not known."""
    # random projection planes
    num_of_los = 100
    planes = np.random.rand(num_of_los, 3)
    planes = a/np.linalg.norm(a)

    # sort particles into radii bins
    indx = np.digitize(np.linalg.norm(pos), bins, right=True)
    
    beta = np.zeros(len(bins) - 1)
    for b in bins:
        # select particles in shell
        pos_shell = pos[indx == b]
        vel_shell = vel[indx == b]

        for p in range(num_of_los):
            vel_shell_proj = vel_shell*planes[p]
            sigma = np.std(vel_shell_proj)
        
        # Calculate Velocity Dispersion
        sigma = [np.std(vlos[m]) for m in range(len(vlos))]
        sigma_median[sbh_indx] = np.median(sigma)
        sigma_std[sbh_indx] = np.std(sigma)
        sigma_70perc[sbh_indx] = np.percentile(sigma, 70)
    
    return sigma_median, sigma_std, sigma_70perc




