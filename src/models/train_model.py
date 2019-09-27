import os, time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Any, Callable

import numpy as np

import pickle
from comet_ml import Experiment

from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    mean_squared_error,
    precision_score,
    precision_recall_fscore_support,
)

from GaHaCo.src.utils.datautils import (
    get_data,
    balance_dataset,
    find_transition_regions,
)
from GaHaCo.src.utils.config import load_config
from GaHaCo.src.visualization import visualize

import logging

# -----------------------------------------------------------------------------
# Loggings
# -----------------------------------------------------------------------------
tag_datetime = datetime.now().strftime("%H%M_%d%m%Y")

experiment = Experiment(
    api_key="VNQSdbR1pw33EkuHbUsGUSZWr", project_name="general", workspace="florpi"
)

logging.basicConfig(
    filename="../../models/rnf/log_%s.log" % tag_datetime,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)

# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------
def main(model, label, sampling, PCA):

	logging.info("")
	logging.info("GROWING TREES")
	logging.info("")
	
    # ML-model settings
    config_file_path = "../../models/rnf/config_%s.json" % model
    config = load_config(config_file_path=config_file_path)

    # -------------------------------------------------------------------------
    # Load and prepare datasets
    # -------------------------------------------------------------------------

    # Load dataset
    output_file = "merged_dataframe.h5"
    data_path = "/cosma6/data/dp004/dc-cues1/tng_dataframes/"
    hdf5_filename = data_path + output_file
    train, test, test_pos_hydro = get_data(hdf5_filename, label)

    # Prepare datasets
    ## Balance training set in the transition region
    center_transition, end_transition = find_transition_regions(train)

    """
	ex.log_scalar(
		"The labels before balancing are as follows:", train.labels.value_counts()
	)
	"""
    train = balance_dataset(train, center_transition, end_transition, sampling)
    """
	ex.log_scalar(
		"The labels after balancing are as follows:\n a)",
		train[train.M200c < center_transition].labels.value_counts(),
	)
	ex.log_scalar(
		"b)",
		train[
			(train.M200c > center_transition) & (train.M200c < end_transition)
		].labels.value_counts(),
	)
	"""

    train_features = train.drop(columns="labels")
    train_labels = train["labels"]

    test_features = test.drop(columns="labels")
    test_labels = test["labels"]

    ## Standarize features
    scaler = StandardScaler()
    scaler.fit(train_features)
    train_features = scaler.transform(train_features)
    test_features = scaler.transform(test_features)

    if PCA is True:
        # n_comp = 7
        # pca = PCA(n_components=n_comp)
        # pca = PCA().fit(train_features)
        pca_data = PCA().fit_transform(train_features)
        pca_inv_data = PCA().inverse_transform(np.eye(len(feature_names)))

    # -------------------------------------------------------------------------
    # Set-up and Run random-forest (RNF) model
    # -------------------------------------------------------------------------

    rf = RandomForestClassifier(**config["model"]["parameters"])
    rf.fit(train_features, train_labels)

    # Run RNF
    test_pred = rf.predict(test_features)

    # Save results
    precision = precision_score(test_labels, test_pred)
    metrics = {"precision": precision}

    experiment.log_metrics(metrics)

    # -------------------------------------------------------------------------
    # Save output's visualizations
    # -------------------------------------------------------------------------

    visualize.plot_confusion_matrix(
        test_labels,
        test_pred,
        classes=["Dark", "Luminous"],
        normalize=True,
        experiment=experiment,
    )

	#visualize.plot_tpcf(pred_test_positions, label_test_positions, 
	#		experiment = experiment)

    label_test_positions = test_pos_hydro[test.labels, :]
    pred_test_positions = (np.vstack([test.x_dmo, test.y_dmo, test.z_dmo]).T)[
        test_pred, :
    ]

    visualize.plot_tpcf(
        pred_test_positions, label_test_positions, experiment=experiment
    )


# TODO: change to autoconfig
if __name__ == "__main__":

    main("rnf", "dark_or_light", "downsample", False)
