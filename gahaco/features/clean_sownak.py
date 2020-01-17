import pandas as pd
import h5py
import numpy as np

data_path = '/cosma5/data/dp004/hvrn44/HOD/'
matching_file = 'MatchedHaloes_L205n2500.dat'
mergertree_file = 'MergerTree_L205n2500TNG_DM_ext_New.hdf5'
output_path = '/cosma7/data/dp004/dc-cues1/tng_dataframes/'
output_file = 'TNG300Dark_Hydro_MergerTree.hdf5'

# ------------------ Halo matching between dmo and hydro simulations

matching_df = pd.read_csv(data_path + matching_file,
                         delimiter = ' ', skiprows = 1,
                names = ['ID_DMO', 'ID_HYDRO', 'M200_DMO', 'M200_HYDRO'])


# ----------- Read in properties from the merger trees 
with h5py.File(data_path + mergertree_file, 'r') as hf:
    formation_time = hf['Haloes']['z0p50'][:]
    n_mergers = hf['Haloes']['NMerg'][:]
    mass_peak = hf['Haloes']['Mpeak'][:]
    vpeak = hf['Haloes']['Vpeak'][:]
    mergertree_ids = hf['Haloes']['Index'][:]

mergertree_data = np.vstack([mergertree_ids, formation_time, n_mergers,
                            mass_peak, vpeak]).T

mergertree_df = pd.DataFrame(data = mergertree_data, 
                columns = ['ID_DMO', 'Formation Time', 'Nmergers','MassPeak', 'vpeak'])

mergertree_df = pd.merge(matching_df, mergertree_df, on = ['ID_DMO'], how = 'inner')

print(mergertree_df.head(3))

mergertree_df.to_hdf(output_path+ output_file, key = 'df', mode = 'w')
