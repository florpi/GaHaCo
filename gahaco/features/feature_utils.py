import numpy as np
import h5py
import pandas as pd

from typing import Any, Callable
from scipy.stats import binned_statistic
from scipy.interpolate import interp1d

from sklearn.utils import resample

from imblearn.over_sampling import SMOTE

def get_data(arg_label:str,
        boxsize:int=100,
        path_to_file:str="/cosma7/data/dp004/dc-cues1/tng_dataframes/",
            ):
    """
    """
    filename = f"merged_dataframe_{boxsize}.h5"
    hdf5_filename = path_to_file + filename
    df = pd.read_hdf(hdf5_filename, key="df", mode="r")
    df = df.fillna(-9999.)

    ids = df.ID_DMO
    drop_list=["N_gals", "M_stars_central", "total_M_stars",
            "x_hydro", "y_hydro", "z_hydro", 
            "x_dmo", "y_dmo", "z_dmo",
            "M200_HYDRO", "ID_HYDRO", "ID_DMO",
            "Group_R_Crit200", "CentralVmax", #"m2500c",
            "vrms_2500c", "vrms_200c", "vrms_std_2500c",
            "CentralMassInMaxRad",
            "displacement",
            #"env_5", "env_10",
            ]
    # Chose label
    if arg_label == "dark_or_light":
        df["labels"] = df.N_gals > 0
        df = df.drop(columns=drop_list)
    elif arg_label == "nr_of_satellites":
        df["labels"] = df.N_gals - 1
        df = df[df.N_gals > 1]
        df = df.drop(columns=drop_list)
    elif arg_label == "stellar_mass":
        df["labels"] = np.log10(df.M_stars_central)
        df["labels"] = df["labels"].replace([-np.inf, np.inf], 0.)
        df = df.drop(columns=drop_list)
    elif arg_label == "both":
        df["labels"] = df.N_gals > 0

    #keep_list = [
    #    "Formation Time", "CentralVelDisp", "M200_DMO", "labels", "CentralMass",
    #    "env_10"
    #]

    #df = df[keep_list]

    return df.drop(columns="labels"), df.labels


def load_positions(test_idx = None,
        path_to_file:str="/cosma7/data/dp004/dc-cues1/tng_dataframes/",
        boxsize:int=100
        ):
    filename = f"merged_dataframe_{int(boxsize)}.h5"
    hdf5_filename = path_to_file + filename
    df = pd.read_hdf(hdf5_filename, key="df", mode="r")
    if test_idx is not None:
        df=df.iloc[test_idx]
    hydro_pos = np.vstack([df.x_hydro, df.y_hydro, df.z_hydro]).T
    dmo_pos = np.vstack([df.x_dmo, df.y_dmo, df.z_dmo]).T
    return hydro_pos, dmo_pos


def _find_transition_regions(df_features: pd.DataFrame, n_centrals):
    """

    Function to find two masses: where half the haloes are luminous, and where all haloes are luminous

    Args:
        df: dataframe containing masses and wheather luminous or dark
    Returns:
        mass_center: mass at which half of the haloes are luminous.
        mass_end: mass at which 100% of haloes are luminous.

    """
    nbins = 15
    m200c = 10**df_features.M200c
    bins = np.logspace(np.log10(np.min(m200c)), 12.5, nbins + 1)

    nluminous, mass_edges, _ = binned_statistic(
        m200c, n_centrals, statistic="mean", bins=bins
    )

    interpolator = interp1d(nluminous, (mass_edges[1:] + mass_edges[:-1]) / 2.0)

    mass_center = interpolator(0.5)
    mass_end = ((mass_edges[1:] + mass_edges[:-1]) / 2.0)[nluminous > 0.99][0]
    return np.log10(mass_center), np.log10(mass_end)


def balance_dataset(df_features, df_labels, sampler, split='mass'):

    if split == 'mass':
        df_features_resampled, df_labels_resampled=_balance_mass_split(df_features, 
                                                    df_labels, sampler)
    else:
        df_features_resampled, df_labels_resampled=_balance(df_features, df_labels, sampler)
    return df_features_resampled, df_labels_resampled


def _balance(df_features, df_labels, sampler):
    sampler_ = sampler(random_state=42) 
    features_resampled, labels_resampled = sampler_.fit_sample(df_features, df_labels)
    df_features_resampled = pd.DataFrame(data=features_resampled,
                                    columns=df_features.columns)
    df_labels_resampled= pd.Series(data=labels_resampled)

    return df_features_resampled, df_labels_resampled 


def _balance_mass_split(
    df_features, df_labels, sampler
):
    center_transition, end_transition = _find_transition_regions(df_features, df_labels)
    df_left_transition_feats, df_left_transition_labels = _balance_df_given_mass(
        df_features, df_labels, 0.0, center_transition, sampler 
    )
    df_right_transition_feats, df_right_transition_labels = _balance_df_given_mass(
            df_features, df_labels, center_transition, 15, sampler
        )
    df_features = pd.concat([df_left_transition_feats, df_right_transition_feats])
    df_labels = pd.concat([df_left_transition_labels, df_right_transition_labels])

    return df_features, df_labels


def _balance_df_given_mass(
    df_features, df_labels, minimum_mass, maximum_mass, sampler
):
    """
    internal function indicated by leading _
    """
    mass_threshold = (df_features.M200c > minimum_mass) & (df_features.M200c < maximum_mass)

    df_M = df_features[mass_threshold]
    df_M_labels = df_labels[mass_threshold]

    df_features_resampled, df_labels_resampled = _balance(df_M, df_M_labels, sampler)
    return df_features_resampled, df_labels_resampled
