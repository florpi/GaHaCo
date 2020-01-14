import pandas as pd
import h5py
import numpy as np


additional_data_path = '/cosma5/data/dp004/hvrn44/HOD/'
output_file = 'merged_dataframe.h5'
data_path = '/cosma7/data/dp004/dc-cues1/tng_dataframes/'
anisotropy_path = "/cosma7/data/dp004/dc-beck3/vel_ani_param.hdf5"
dmo_file =  'dmo_halos.hdf5'
hydro_file = 'hydro_galaxies.hdf5'
matching_file = 'MatchedHaloes_L205n2500.dat'
additional_properties_file = 'HaloProfiles_DMO_z0.00_ext.hdf5'
mergertree_file = 'MergerTree_L205n2500TNG_DM_ext_New.hdf5'
halo_particles_file = 'halo_particle_summary.hdf5'
inner_file = 'TNG300dark_halo_particle_ss_r2500c.hdf5'
halo_mass_cut = 1.e11

# ------------------ Halo matching between dmo and hydro simulations

matching_df = pd.read_csv(additional_data_path + matching_file,
                         delimiter = ' ', skiprows = 1,
                names = ['ID_DMO', 'ID_HYDRO', 'M200_DMO', 'M200_HYDRO'])

# Apply mass cut

mass_matching_df = matching_df.loc[matching_df['M200_HYDRO'] > halo_mass_cut]

# ----------- Read in additionally halo properties from the dmo simulation

with h5py.File(additional_data_path + additional_properties_file,  'r') as hf:
        
    mass = hf['Haloes']['M200'][:]
    cnfw = hf['Haloes']['Cnfw'][:]
    rmax = hf['Haloes']['Rmax'][:]
    r200c = hf['Haloes']['R200'][:]
    rhosnfw = hf['Haloes']['Rhosnfw'][:]
    properties_ids = hf['Haloes']['GroupNumber'][:]

properties = np.vstack([properties_ids,mass, rmax,
                                     np.log10(r200c), cnfw, rhosnfw]).T


properties_df = pd.DataFrame(data = properties,
                             columns = ['ID_DMO', 'M200c', 'Rmax',
                                        'R200c', 'Cnfw', 'Rhosnfw'])

merged_matching_df = pd.merge(mass_matching_df, properties_df, on = ['ID_DMO'], how = 'inner')

 
# ----------- Read in properties from the merger trees 
with h5py.File(additional_data_path + mergertree_file, 'r') as hf:

    formation_time = hf['Haloes']['z0p50'][:]
    n_mergers = hf['Haloes']['NMerg'][:]
    mass_peak = hf['Haloes']['Mpeak'][:]
    vpeak = hf['Haloes']['Vpeak'][:]
    mergertree_ids = hf['Haloes']['Index'][:]

mergertree_data = np.vstack([mergertree_ids, formation_time, n_mergers,
                            np.log10(mass_peak), vpeak]).T

mergertree_df = pd.DataFrame(data = mergertree_data, 
                columns = ['ID_DMO', 'Formation Time', 'Nmergers','MassPeak', 'vpeak'])

merged_tree_df = pd.merge(merged_matching_df, mergertree_df, on = ['ID_DMO'], how = 'inner')

# ----------- Read in properties from tng dmo

dmo_df = pd.read_hdf(data_path + dmo_file)

dmo_merged_df = pd.merge(merged_tree_df, dmo_df, on = ['ID_DMO'], how = 'inner')

# Check the merged haloes are the correct ones
np.testing.assert_allclose(dmo_merged_df.M200_DMO, 10**dmo_merged_df.Group_M_Crit200, rtol = 1e-3)
dmo_merged_df = dmo_merged_df.drop(columns = ['Group_M_Crit200'])
np.testing.assert_allclose(10**dmo_merged_df.R200c, dmo_merged_df.Group_R_Crit200, rtol = 1e-3)
dmo_merged_df = dmo_merged_df.drop(columns = ['Group_R_Crit200'])

# ----------- Read in properties from tng particle data
particle_df = pd.read_hdf(data_path + halo_particles_file)
particle_df = particle_df.drop(columns = ['m200c'])

dmo_merged_df = pd.merge(dmo_merged_df, particle_df, on=['ID_DMO'])
# Add velocity anisotropy

anisotropy_df = pd.read_hdf(anisotropy_path)
anisotropy_df = anisotropy_df.drop(columns=['index'])

dmo_merged_df = pd.merge(dmo_merged_df, anisotropy_df, on=['ID_DMO'])

# Add r2500c values
inner_df = pd.read_hdf(data_path + inner_file)
np.testing.assert_allclose(
    inner_df.m200c, 10**dmo_merged_df.Group_M_Crit200, rtol = 1e-3
)
inner_df = inner_df.drop(columns = ['m200c'])
inner_df = inner_df["m2500c"].apply(np.log10)

dmo_merged_df = pd.merge(dmo_merged_df, inner_df, on=['ID_DMO'])

# ----------- Read in properties from tng hydro 
hydro_df = pd.read_hdf(data_path + hydro_file)

hydro_merged_df = pd.merge(dmo_merged_df, hydro_df, on = ['ID_HYDRO'], how = 'inner', suffixes = ('_dmo', '_hydro'))

np.testing.assert_allclose(hydro_merged_df.M200_HYDRO, 10**hydro_merged_df.Group_M_Crit200, rtol = 1e-3)
hydro_merged_df = hydro_merged_df.drop(columns = ['Group_M_Crit200', 'M200_DMO'])
hydro_merged_df.M200c = np.log10(hydro_merged_df.M200c)

# Since matching is not 1-to-1, sum all the galaxies
n_gals_total = hydro_merged_df.groupby('ID_DMO')['N_gals'].sum()
m_stars_total = hydro_merged_df.groupby('ID_DMO')['M_stars'].sum()
# Drop duplicates
no_duplicates_df = hydro_merged_df.drop_duplicates(subset = 'ID_DMO', keep = 'last')
# Add column with total galaxies
no_duplicates_df = pd.merge(no_duplicates_df, n_gals_total.to_frame('N_gals_total').reset_index())
no_duplicates_df = pd.merge(no_duplicates_df, m_stars_total.to_frame('M_stars_total').reset_index())
no_duplicates_df = no_duplicates_df.drop(columns = ['N_gals', 'M_stars'])
no_duplicates_df.rename(columns = {'N_gals_total':'N_gals', 'M_stars_total':'M_stars'}, inplace = True)

# Save final dataframe!

print(f'Saving final dataframe into {data_path + output_file}')

no_duplicates_df.to_hdf(data_path + output_file, key = 'df', mode = 'w')
