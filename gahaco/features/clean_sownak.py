import pandas as pd
import h5py
import numpy as np

tng = 100

if tng==100:
    extension = 'L75n1820' 
elif tng == 300:
    extension = 'L205n2500'
else:
    extension = 'NotFound'

data_path = '/cosma5/data/dp004/hvrn44/HOD/'
if tng==100:
    matching_file = f'MatchedHaloes_{extension}TNG.dat'
else:
    matching_file = f'MatchedHaloes_{extension}.dat'
mergertree_file = f'MergerTree_{extension}TNG_DM_ext_New.hdf5'
output_path = '/cosma7/data/dp004/dc-cues1/tng_dataframes/'
output_file = f'TNG{tng}Dark_Hydro_MergerTree.hdf5'

# ------------------ Halo matching between dmo and hydro simulations

matching_df = pd.read_csv(data_path + matching_file,
                         delimiter = ' ', skiprows = 1,
                names = ['ID_DMO', 'ID_HYDRO', 'M200_DMO', 'M200_HYDRO'])


# ----------- Read in properties from the merger trees 
with h5py.File(data_path + mergertree_file, 'r') as hf:
    formation_time = hf['Haloes']['z0p50'][:]
    if tng == 300:
        n_mergers = hf['Haloes']['NMerg'][:]
        #mass_peak = hf['Haloes']['Mpeak'][:]
        #vpeak = hf['Haloes']['Vpeak'][:]
    mergertree_ids = hf['Haloes']['Index'][:]

if tng==300:
    mergertree_data = np.vstack([mergertree_ids, formation_time, n_mergers,]).T
                            #mass_peak, vpeak]).T
else:
    mergertree_data = np.vstack([mergertree_ids, formation_time]).T


if tng==300:
    mergertree_df = pd.DataFrame(data = mergertree_data, 
                columns = ['ID_DMO', 'Formation Time', 'Nmergers','MassPeak', 'vpeak'])
else:
    mergertree_df = pd.DataFrame(data = mergertree_data, 
                columns = ['ID_DMO', 'Formation Time' ])


mergertree_df = pd.merge(matching_df, mergertree_df, on = ['ID_DMO'], how = 'inner')

print(mergertree_df.head(3))

mergertree_df.to_hdf(output_path+ output_file, key = 'df', mode = 'w')
