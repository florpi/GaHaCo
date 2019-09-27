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
from GaHaCo.src.utils.optimize import feature_optimization
from GaHaCo.src.utils.config import load_config
from GaHaCo.src.visualization import visualize
from GaHaCo.src.models.predict_model import prediction

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
def main():

    logging.info("")
    logging.info("GROWING TREES")
    logging.info("")

    # ML-model settings
    config_file_path = "../../models/rnf/config_%s.json" % model
    config = load_config(config_file_path=config_file_path)
    print("+++ config :", config)

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
    train = balance_dataset(train, center_transition, end_transition, sampling)

    train_features = train.drop(columns="labels")
    train_labels = train["labels"]

    test_features = test.drop(columns="labels")
    test_labels = test["labels"]

    ## Standarize features
    scaler = StandardScaler()
    scaler.fit(train_features)
    train_features = scaler.transform(train_features)
    test_features = scaler.transform(test_features)
    # convert pd.dataframe to np.ndarray
    train_labels = train_labels.values
    test_labels = test_labels.values

    train = {"features": train_features, "labels": train_labels}
    test = {"features": test_features, "labels": test_labels}

    if "feature_optimization" in config.keys():
        # Perform feature optimization
        train["features"], test["features"] = feature_optimization(
            train, test, config["feature_optimization"], experiment=experiment
        )

    # -------------------------------------------------------------------------
    # Set-up and Run inference model
	# -------------------------------------------------------------------------

	test_pred = prediction(train, test, config["model"])

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

    # visualize.plot_tpcf(pred_test_positions, label_test_positions,
    # 		experiment = experiment)

    label_test_positions = test_pos_hydro[test.labels, :]
    pred_test_positions = (np.vstack([test.x_dmo, test.y_dmo, test.z_dmo]).T)[
        test_pred, :
    ]

    visualize.plot_tpcf(
        pred_test_positions, label_test_positions, experiment=experiment
    )


# TODO: change to autoconfig
if __name__ == "__main__":

    main()
