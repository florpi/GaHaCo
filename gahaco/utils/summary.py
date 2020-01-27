import numpy as np
from sklearn.metrics import confusion_matrix

from gahaco.models import hod
from gahaco.utils.tpcf import compute_tpcf

def hod_stellar_mass_summary(
        m200c_train,
        m200c_test,
        stellar_mass_train,
        stellar_mass_test,
        stellar_mass_thresholds, 
        dmo_pos_test,
        boxsize):

    halo_occs, hod_cms, hod_tpcfs = [],[],[]
    for threshold in stellar_mass_thresholds:
        n_gals_train = stellar_mass_train > threshold 
        n_gals_test = stellar_mass_test > threshold 
        halo_occ = hod.HOD(m200c_train, n_gals_train)
        n_gals_test_hod = halo_occ.populate_centrals(m200c_test)
        cm = confusion_matrix(n_gals_test, n_gals_test_hod)
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        hod_cms.append(cm)
        halo_occs.append(halo_occ)
        hod_tpcfs.append(compute_tpcf(dmo_pos_test[n_gals_test_hod > 0],
            boxsize=boxsize)[1])

    return halo_occs, hod_cms, hod_tpcfs

def hod_summary(m200c_train, m200c_test, n_gals_train, n_gals_test, dmo_pos_test, boxsize):
        halo_occ = hod.HOD(m200c_train, n_gals_train)
        n_gals_test_hod = halo_occ.populate_centrals(m200c_test)

        cm = confusion_matrix(n_gals_test, n_gals_test_hod)
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        hod_tpcf = compute_tpcf(dmo_pos_test[n_gals_test_hod > 0],
                boxsize=boxsize)[1]

        return halo_occ, cm, hod_tpcf

 
def model_stellar_mass_summary(stellar_mass_test, 
        stellar_mass_test_pred, stellar_mass_thresholds,
        dmo_pos_test,
        boxsize):

    cms, pred_tpcf = [],[]
    for threshold in stellar_mass_thresholds:
        n_gals_test = stellar_mass_test > threshold 
        n_gals_test_pred = stellar_mass_test_pred > threshold 
        cm = confusion_matrix(n_gals_test, n_gals_test_pred)
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        cms.append(cm)
        pred_tpcf.append(compute_tpcf(dmo_pos_test[n_gals_test_pred > 0],
            boxsize=boxsize)[1])

    return cms, pred_tpcf 


def model_summary(n_gals_test, n_gals_pred, dmo_pos_test, boxsize):
    
    cm = confusion_matrix(n_gals_test, n_gals_pred)
    cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    pred_tpcf = compute_tpcf(dmo_pos_test[n_gals_pred>0],
            boxsize=boxsize)[1]
    return cm, pred_tpcf

def hydro_stellar_mass_summary(hydro_pos_test, stellar_mass_test, stellar_mass_thresholds, boxsize):
    hydro_tpcf = []
    for threshold in stellar_mass_thresholds:
        n_gals_test = stellar_mass_test > threshold 
        hydro_tpcf.append(compute_tpcf(hydro_pos_test[n_gals_test > 0],
            boxsize=boxsize)[1])
    r_c = compute_tpcf(hydro_pos_test[n_gals_test > 0],
            boxsize=boxsize)[0]

    return r_c, hydro_tpcf 

def hydro_summary(hydro_pos_test, n_gals_test, boxsize):
    r_c, hydro_tpcf = compute_tpcf(test_hydro_pos[y_test>0],
            boxsize=boxsize)
    return r_c, hydro_tpcf


