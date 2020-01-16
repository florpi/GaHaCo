import sys

sys.path.insert(0,"/cosma/home/dp004/dc-cues1/arepo_hdf5_library")
import read_hdf5
import numpy as np
from scipy.spatial import distance as fast_distance
from scipy.optimize import curve_fit
import h5py
import pandas as pd
import argparse
from multiprocessing import Pool
from functools import partial
from scipy.spatial import cKDTree

class Catalog:
    '''

    Class to describe a catalog and its methods


    '''
    def __init__(self, 
            h5_dir: str, 
            snapnum: int,
            n_cores:int=1):
        '''
        Args:
            h5_dir: directory containing tng data
            snapnum: Snapshot number to read
        '''

        self.snapnum = snapnum

        self.snapshot = read_hdf5.snapshot(snapnum, h5_dir)

        # Useful definitions
        self.dm = 1
        self.stars = 4
        self.dm_particle_mass = self.snapshot.header.massarr[self.dm] * 1.0e10
        self.output_dir = '/cosma7/data/dp004/dc-cues1/tng_dataframes/'
        self.n_cores = n_cores

    def load_inmidiate_features(self, 
            group_feature_list: list, 
            sub_feature_list: list):
        """
        Loads features already computed by SUBFIND 
        + Bullock spin parameter (http://iopscience.iop.org/article/10.1086/321477/fulltext/52951.text.html)

        Args:
            group_feature_list: list of group features to load and save as attributes of the class
            sub_feature_list: list of subgroup features to load and save as attributes of the class

        """

        for feature in group_feature_list:
            value = self.snapshot.cat[feature][self.halo_mass_cut]
            if ('Crit200' in feature) or ('Mass' in feature):
                value *= self.snapshot.header.hubble
            setattr(self, feature, value)

        self.firstsub = (self.GroupFirstSub).astype(int)

        self.v200c = np.sqrt(self.snapshot.const.G * self.Group_M_Crit200/ self.Group_R_Crit200/1000.) * self.snapshot.const.Mpc / 1000. 

        for feature in sub_feature_list:
            value = self.snapshot.cat[feature][self.firstsub]
            if ('Crit200' in feature) or ('Mass' in feature):
                value *= self.snapshot.header.hubble
            setattr(self, feature.replace('Subhalo', ''), value)

        self.Spin= (np.linalg.norm(self.Spin, axis=1)/3.) / np.sqrt(2) / self.Group_R_Crit200/self.v200c

        self.bound_mass = self.MassType[:, self.dm] 
        self.total_mass = self.GroupMassType[:, self.dm]


    def compute_x_offset(self):
        """
        Computes relaxadness parameter, which is the offset between the halo center of mass and its most bound particle 
        position in units of r200c
        http://arxiv.org/abs/0706.2919

        """

        self.x_offset = self.periodic_distance(self.GroupCM, self.GroupPos) / self.Group_R_Crit200

    def compute_fsub_unbound(self):
        """

        Computes another measure of how relaxed is the halo, defined as the ration between mass bound to the halo and 
        mass belonging to its FoF group

        """

        self.fsub_unbound = 1.0 - self.bound_mass / self.total_mass

    def closest_node(self, node, nodes):
            return fast_distance.cdist([node], nodes).argmin()


    def Environment_subhalos(self, r_outer):
        '''
        
        Measure of a halo's environment. Computes the difference in subhalo masses in a sphere of radius r_outer and a sphere
        of radius r200c of the halo.
        https://arxiv.org/abs/1911.02610

        Args:
            r_outer: radius of the outer sphere (in Mpc/h).

        Returns:
            env_per_halo: array of len number of halos, with the measure of environemnt 

        '''

        # Load subhalos
        r_outer = 1000. * r_outer # to kpc/h
        subhalo_pos = self.snapshot.cat['SubhaloPos'][:]
        subhalo_mass = self.snapshot.cat['SubhaloMass'][:]
        # Construct tree with all subhalos
        tree=cKDTree(subhalo_pos, boxsize=self.boxsize*1000.)
        # For each halo, query around
        r_inner = self.Group_R_Crit200 
        outer_ids = tree.query_ball_point(self.GroupPos, r_outer)
        env_per_halo = np.zeros((len(self.GroupPos)))
        for halo in range(len(self.GroupPos)):
            inner_ids = tree.query_ball_point(self.GroupPos[halo,:], r_inner[halo])
            env_per_halo[halo] = np.sum(subhalo_mass[outer_ids[halo]]) - np.sum(subhalo_mass[inner_ids])
        return env_per_halo

    def Environment_haas(self, f: float, halo_idx: int):
        """

        Measure of environment that is not correlated with host halo mass http://arxiv.org/abs/1103.0547.
        Outputs: haas_env, distance to the closest neighbor with a mass larger than f * m200c, divided by its r200c 

        Args: 
            f: threshold to select minimum mass of neighbor to consider.

        """

        #haas_env = np.zeros(self.N_halos)

        #for i in range(self.N_halos):
        halopos_exclude = np.delete(self.GroupPos, halo_idx, axis=0)
        m200c_exclude = np.delete(self.Group_M_Crit200, halo_idx)

        halopos_neighbors = halopos_exclude[(m200c_exclude >= f * self.Group_M_Crit200[halo_idx])]
        if halopos_neighbors.shape[0] == 0:
            return -1.
            #haas_env[i] = -1.0
        index_closest = self.closest_node(self.GroupPos[halo_idx], halopos_neighbors)
        distance_fneigh = np.linalg.norm(
            self.GroupPos[halo_idx] - halopos_neighbors[index_closest]
        )

        r200c_exclude = np.delete(self.Group_R_Crit200, halo_idx)
        r200c_neighbor = r200c_exclude[(m200c_exclude >= f * self.Group_M_Crit200[halo_idx])][
            index_closest
        ]
        haas_env = distance_fneigh / r200c_neighbor
        return haas_env


    def total_fsub(self):
        """

        Fraction of mass bound to substructure compared to the halo mass.
        Outputs: fsub, ratio of M_fof/M_bound

        """
        fsub = np.zeros((self.N_halos))
        for i in range(self.N_halos):
            fsub[i] = (
                np.sum(
                    self.snapshot.cat["MassType"][
                        self.subhalo_offset[i]
                        + 1 : self.subhalo_offset[i]
                        + self.N_particles[i],
                        :,
                    ]
                )
                / self.m200c[i]
            )

        return fsub

    def periodic_distance(self, a: np.ndarray, b: np.ndarray) -> np.array:
        """

        Computes distance between vectors a and b in a periodic box
        Args:
            a: first array.
            b: second array.
        Returns:
            dists, distance once periodic boundary conditions have been applied

        """

        bounds = self.boxsize * np.ones(3)

        min_dists = np.min(np.dstack(((a - b) % bounds, (b - a) % bounds)), axis=2)
        dists = np.sqrt(np.sum(min_dists ** 2, axis=1))
        return dists

    def halo_shape(self):
        """

        Describes the shape of the halo
        http://arxiv.org/abs/1611.07991

        """ 
        inner = 0.15 # 0.15 *r200c (inner halo)
        outer = 1.
        self.inner_q = np.zeros(self.N_halos)
        self.inner_s = np.zeros(self.N_halos)
        self.outer_q = np.zeros(self.N_halos)
        self.outer_s = np.zeros(self.N_halos)
        for i in range(self.N_halos):
            coordinates_halo = self.coordinates[self.group_offset[i] : self.group_offset[i] + self.N_particles[i],:]
            distance = (coordinates_halo - self.GroupPos[i])/self.r200c[i]
            self.inner_q[i],self.inner_s[i], _, _  = ellipsoid.ellipsoidfit(distance,\
                    self.r200c[i], 0,inner,weighted=True)
            self.outer_q[i],self.outer_s[i], _, _  = ellipsoid.ellipsoidfit(distance,\
                    self.r200c[i], 0,outer,weighted=True)


    def save_features(self, 
            output_filename: str, 
            features_to_save: list):
        '''
        Save given features to hdf5 

        Args:
            output_filename: file to save hdf5 file.
            features_to_save: list of feature names to save into file.

        '''

        print(f'Saving their properties into {self.output_dir + output_filename}')

        if 'GroupPos' in features_to_save:
            remove_grouppos = True
        else:
            remove_grouppos = False

        feature_list = []
        for feature in features_to_save:
            if feature != 'GroupPos':
                feature_list.append(getattr(self, feature))

        feature_list = np.asarray(feature_list).T
        features_to_save.remove('GroupPos') if 'GroupPos' in features_to_save else None

        df = pd.DataFrame( data = feature_list,
                columns = features_to_save)

        if remove_grouppos:
            df['x'] = self.GroupPos[:,0]/1000. # To Mpc/h
            df['y'] = self.GroupPos[:,1]/1000.
            df['z'] = self.GroupPos[:,2]/1000.

        df.to_hdf(self.output_dir + output_filename, key = 'df', mode = 'w')




class HaloCatalog(Catalog):

    def __init__(self, n_cores=1):
        """
        Class to read halo catalogs from simulation

        """

        # Read snapshot
        h5_dir = "/cosma7/data/TNG/TNG300-1-Dark/"
        super().__init__(h5_dir, 99, n_cores)
        self.boxsize = self.snapshot.header.boxsize / self.snapshot.header.hubble # kpc
        self.halo_mass_thresh = 5.0e10 

        print("Minimum DM halo mass : %.2E" % self.halo_mass_thresh)

        # Load fields that will be used
        group_properties = [
            "GroupFirstSub",
            "Group_M_Crit200",
            "GroupNsubs",
            "GroupPos",
            "GroupVel",
            "Group_R_Crit200",
            "GroupMassType",
        ]

        sub_properties = [
            "SubhaloMassType",
            "SubhaloCM",
            "GroupCM",
            "SubhaloMass",
            "SubhaloSpin",
            "SubhaloVelDisp",
            "SubhaloVmax",
            "SubhaloPos",
            "SubhaloMassInMaxRad",
            "SubhaloHalfmassRad",
        ]

        self.snapshot.group_catalog(group_properties + sub_properties)

        self.N_subhalos = (self.snapshot.cat["GroupNsubs"]).astype(np.int64)
        # Get only resolved halos
        self.halo_mass_cut = (
                self.snapshot.cat["Group_M_Crit200"][:] * self.snapshot.header.hubble > self.halo_mass_thresh
        )
        # Save IDs of haloes
        self.ID_DMO = np.arange(0, len(self.halo_mass_cut))
        self.ID_DMO = self.ID_DMO[self.halo_mass_cut]

        self.subhalo_offset = (np.cumsum(self.N_subhalos) - self.N_subhalos).astype(
            np.int64
        )
        self.subhalo_offset = self.subhalo_offset[self.halo_mass_cut]

        self.N_subhalos = self.N_subhalos[self.halo_mass_cut]

        self.N_halos = self.N_subhalos.shape[0]
        print("%d resolved halos found." % self.N_halos)

        self.load_inmidiate_features(group_properties, sub_properties)

        self.compute_fsub_unbound()
        self.compute_x_offset()

class GalaxyCatalog(Catalog):
    def __init__(self, snapnum=99):
        """
        Class to read galaxy catalogs from simulation

        """
        # Read snapshot

        h5_dir = "/cosma7/data/TNG/TNG300-1/"
        super().__init__(h5_dir, 99)
        self.boxsize = self.snapshot.header.boxsize / self.snapshot.header.hubble # kpc

        self.stellar_mass_thresh = 1.0e9 
        # Load fields that will be used
        group_properties = [
            "Group_M_Crit200",
            "GroupMassType",
            "GroupNsubs",
            "GroupPos"
        ]

        sub_properties = [
            "SubhaloMassType",
            "SubhaloPos",
        ]

        self.snapshot.group_catalog(group_properties + sub_properties)


        self.halo_mass_cut = (
                self.snapshot.cat["Group_M_Crit200"][:] * self.snapshot.header.hubble > 0. 
        )

        N_halos_all = self.snapshot.cat['Group_M_Crit200'].shape[0]

        self.ID_HYDRO = np.arange(0, N_halos_all)
        self.ID_HYDRO = self.ID_HYDRO[self.halo_mass_cut]

        self.N_halos= (self.snapshot.cat['Group_M_Crit200'])[self.halo_mass_cut].shape[0]

        self.N_subhalos = (self.snapshot.cat["GroupNsubs"]).astype(np.int64)
        self.subhalo_offset = (np.cumsum(self.N_subhalos) - self.N_subhalos).astype(
            np.int64
        )
        self.subhalo_offset = self.subhalo_offset[self.halo_mass_cut]
        self.N_subhalos = self.N_subhalos[self.halo_mass_cut]


        self.Group_M_Crit200 = self.snapshot.cat['Group_M_Crit200'][self.halo_mass_cut] * self.snapshot.header.hubble
        self.GroupPos = self.snapshot.cat['GroupPos'][self.halo_mass_cut, :] 
        self.N_gals, self.total_M_stars, self.M_stars_central = self.Number_of_galaxies()
        self.pos_gals = self.galaxy_positions()
        print("%d resolved galaxies found." % np.sum(self.N_gals))

        print("Minimum stellar mass : %.2E" % self.stellar_mass_thresh)

    def Number_of_galaxies(self):
        """

        Given the halo catalog computes the stellar mass of a given halo, and its number of galaxies. 
        The number of galaxies is defined as the number of subhalos that halo has over a given stellar mass 
        defined inside the class

        Returns:
                N_gals: number of galaxies belonging to the halo
                M_stars: mass of the stellar component bound to the halo

        """
        # Subhaloes defined as galaxies with a stellar mass larger than the threshold
        N_gals = np.zeros((self.N_halos), dtype=np.int)
        total_M_stars = np.zeros((self.N_halos), dtype=np.int)
        M_stars_central = np.zeros((self.N_halos), dtype=np.int)
        for i in range(self.N_halos):
            N_gals[i] = np.sum(
                self.snapshot.cat["SubhaloMassType"][
                    self.subhalo_offset[i] : self.subhalo_offset[i]
                    + self.N_subhalos[i],
                    self.stars,
                ]
                > self.stellar_mass_thresh
            )
            total_M_stars[i] = np.sum(
                self.snapshot.cat["SubhaloMassType"][
                    self.subhalo_offset[i] : self.subhalo_offset[i]
                    + self.N_subhalos[i],
                    self.stars,
                ]
            )
            M_stars_central[i] = np.sum(
                    self.snapshot.cat["SubhaloMassType"][
                        self.subhalo_offset[i] : self.subhalo_offset[i] + 1,
                        self.stars,
                        ]
                    )

        return N_gals,total_M_stars,  M_stars_central 

    def galaxy_positions(self):

        gals_offset = np.cumsum(self.N_gals) - self.N_gals
        pos_gals = np.zeros((np.sum(self.N_gals), 3))
        for i in range(self.N_halos):
            first_galaxy = gals_offset[i]
            last_galaxy = gals_offset[i] + self.N_gals[i]
            first_subhalo = self.subhalo_offset[i] 
            last_subhalo = self.subhalo_offset[i] + self.N_subhalos[i]

            luminous_subhalos =  self.snapshot.cat["SubhaloMassType"][first_subhalo:last_subhalo,
                    self.stars] > self.stellar_mass_thresh

            pos_gals[first_galaxy:last_galaxy,:] = self.snapshot.cat['SubhaloPos'][first_subhalo:last_subhalo,:][luminous_subhalos]

        return pos_gals


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Execute in N cores')
    parser.add_argument('--np', dest='n_cpu', type=int, help='number of available cpus')
    args = parser.parse_args()
    print(f'There are {args.n_cpu} cores available')

    halocat = HaloCatalog()

    env_5 = halocat.Environment_subhalos(5)
    halocat.env_5 = np.log10(env_5)
    env_10 = halocat.Environment_subhalos(10)
    halocat.env_10 = np.log10(env_10)

    features_to_save = ['ID_DMO','N_subhalos', 'Group_M_Crit200', 'Group_R_Crit200',
            'VelDisp', 'Vmax', 'Spin', 'fsub_unbound', 'x_offset' , 'GroupPos',
            "HalfmassRad","MassInMaxRad",
            'env_5', 'env_10']
    halocat.save_features('TNG300dark_subfind.hdf5', features_to_save)

    galcat = GalaxyCatalog()
    features_to_save = ['ID_HYDRO','N_gals', 'M_stars_central', 'total_M_stars', 'Group_M_Crit200', 'GroupPos']
    galcat.save_features('TNG300hydro_subfind.hdf5', features_to_save)
    np.save(galcat.output_dir + 'galaxy_positions', galcat.pos_gals)
