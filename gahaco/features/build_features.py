import pandas as pd
import h5py
import numpy as np

boxsize = 300
halo_mass_cut = 1.e11

output_file = "merged_dataframe_v2.h5"
data_path = '/cosma7/data/dp004/dc-cues1/tng_dataframes/'
mergertree_file = data_path + 'TNG%dDark_Hydro_MergerTree.hdf5' % (boxsize)
subfind_dark_file = data_path + 'TNG%ddark_subfind.hdf5' % (boxsize)
subfind_hydro_file = data_path + 'TNG%dhydro_subfind.hdf5' % (boxsize)
particle_outer_file = data_path + 'TNG%ddark_halo_particle_summary_r200c.hdf5' % (boxsize)
particle_inner_file = data_path + 'TNG%ddark_halo_particle_summary_r2500c.hdf5' % (boxsize)

# Read MergerTree Data ----------------------------------------------------------
mt_df = pd.read_hdf(mergertree_file)
dmo_df = mt_df.loc[mt_df['M200_HYDRO'] > halo_mass_cut]

# Read SubFind Data -------------------------------------------------------------
sf_df = pd.read_hdf(subfind_dark_file)
sf_df = sf_df.loc[10**sf_df['Group_M_Crit200'] > halo_mass_cut]
dmo_df = pd.merge(sf_df, dmo_df, on=['ID_DMO'], how='inner')
np.testing.assert_allclose(
    10**dmo_df.Group_M_Crit200, dmo_df.M200_DMO, rtol=1e-3
)
dmo_df = dmo_df.drop(columns = ['Group_M_Crit200'])

# Read Particle Data ------------------------------------------------------------
hubble = 0.6774
# R200c
pouter_df = pd.read_hdf(particle_outer_file)
pouter_df["m200c"] = pouter_df["m200c"].apply(lambda x: x*hubble)
pouter_df = pouter_df.loc[pouter_df['m200c'] > halo_mass_cut]
dmo_df = pd.merge(pouter_df, dmo_df, on=['ID_DMO'], how='inner')
np.testing.assert_allclose(
    dmo_df.m200c, dmo_df.M200_DMO, rtol=1e-3
)
dmo_df = dmo_df.drop(columns = ['m200c'])

# R2500c
pinner_df = pd.read_hdf(particle_outer_file)
pinner_df["m200c"] = pinner_df["m200c"].apply(lambda x: x*hubble)
pinner_df = pinner_df.loc[pinner_df['m200c'] > halo_mass_cut]
dmo_df = pd.merge(pinner_df, dmo_df, on=['ID_DMO'], how='inner')
np.testing.assert_allclose(
    dmo_df.m200c, dmo_df.M200_DMO, rtol=1e-3
)
dmo_df = dmo_df.drop(columns = ['m200c'])

# Read in properties from tng hydro ---------------------------------------------
hydro_df = pd.read_hdf(subfind_hydro_file)

hydro_merged_df = pd.merge(
    dmo_df,
    hydro_df,
    on=['ID_HYDRO'],
    how='inner',
    suffixes=('_dmo', '_hydro')
)
np.testing.assert_allclose(
    hydro_merged_df.M200_HYDRO, 10**hydro_merged_df.Group_M_Crit200, rtol=1e-3
)
hydro_merged_df = hydro_merged_df.drop(columns=['Group_M_Crit200', 'M200_DMO'])
print(hydro_merged_df.columns)

# Since matching is not 1-to-1, sum all the galaxies ----------------------------
# TODO: reconsider the association criteria
n_gals_total = hydro_merged_df.groupby('ID_DMO')['N_gals'].sum()
m_stars_total = hydro_merged_df.groupby('ID_DMO')['M_stars_central'].sum()

# Drop duplicates ---------------------------------------------------------------
no_duplicates_df = hydro_merged_df.drop_duplicates(subset='ID_DMO', keep='last')

# Add column with total galaxies ------------------------------------------------
no_duplicates_df = pd.merge(
    no_duplicates_df, n_gals_total.to_frame('N_gals').reset_index()
)
no_duplicates_df = pd.merge(
    no_duplicates_df, m_stars_total.to_frame('M_stars_central').reset_index()
)

# Save final dataframe!

print(f'Saving final dataframe into {data_path + output_file}')

no_duplicates_df.to_hdf(data_path + output_file, key = 'df', mode = 'w')
