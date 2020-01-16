import pandas as pd
import h5py
import numpy as np

boxsize = 100
halo_mass_cut = 1.e11

data_path = '/cosma7/data/dp004/dc-cues1/tng_dataframes/'
mergertree_file = data_path + 'TNG%dDark_Hydro_MergerTree.hdf5' % (boxsize)
subfind_dark_file = data_path + 'TNG%ddark_subfind.hdf5' % (boxsize)
subfind_hydro_file = data_path + 'TNG%dhydro_subfind.hdf5' % (boxsize)
particle_outer_file = data_path + 'TNG%ddark_particle_ss_r200c.hdf5' % (boxsize)
particle_inner_file = data_path + 'TNG%ddark_particle_ss_r2500c.hdf5' % (boxsize)

# Read MergerTree Data ----------------------------------------------------------
mt_df = pd.read_hdf(mergertree_file)
dmo_df = mt_df.loc[mt_df['M200_HYDRO'] > halo_mass_cut]

# Read SubFind Data -------------------------------------------------------------
sf_df = pd.read_hdf(subfind_dark_file)
dmo_df = pd.merge(sf_df, dmo_df, on = ['ID_DMO'], how = 'inner')

# Check the merged haloes are the correct ones ----------------------------------
np.testing.assert_allclose(
    dmo_merged_df.M200_DMO, dmo_merged_df.Group_M_Crit200, rtol=1e-3
)
dmo_merged_df = dmo_merged_df.drop(columns = ['Group_M_Crit200','Group_R_Crit200'])

# Read Particle Data ------------------------------------------------------------
# R200c
pouter_df = pd.read_hdf(particle_outer_file)
np.testing.assert_allclose(
    pouter_df.m200c, pouter_df.Group_M_Crit200, rtol=1e-3
)
pouter_df = particle_df.drop(columns = ['m200c'])
dmo_df = pd.merge(dmo_df, pouter_df, on=['ID_DMO'])

# R2500c
pinner_df = pd.read_hdf(particle_inner_file)
np.testing.assert_allclose(
    inner_df.m200c, dmo_merged_df.Group_M_Crit200, rtol=1e-3
)
pinner_df = pinner_df.drop(columns = ['m200c'])

dmo_df = pd.merge(dmo_df, pinner_df, on=['ID_DMO'])

# Read in properties from tng hydro ---------------------------------------------
hydro_df = pd.read_hdf(subfind_hydro_file)

hydro_merged_df = pd.merge(
    dmo_merged_df,
    hydro_df,
    on=['ID_HYDRO'],
    how='inner',
    suffixes=('_dmo', '_hydro')
)

np.testing.assert_allclose(
    hydro_merged_df.M200_HYDRO, hydro_merged_df.Group_M_Crit200, rtol=1e-3
)
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
