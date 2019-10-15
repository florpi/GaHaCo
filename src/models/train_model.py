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
from GaHaCo.src.models.fit_model import fit 

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
def main(model: str):

	logging.info("")
	logging.info("GROWING TREES")
	logging.info("")

	# ML-model settings
	config_file_path = "../../models/%s/config_%s.json" % (model, model)
	config = load_config(config_file_path=config_file_path)

	experiment.add_tag(f'preprocessing = {config["sampling"]}')

	# -------------------------------------------------------------------------
	# Load and prepare datasets
	# -------------------------------------------------------------------------

	# Load dataset
	output_file = "merged_dataframe.h5"
	data_path = "/cosma6/data/dp004/dc-cues1/tng_dataframes/"
	hdf5_filename = data_path + output_file
	train_df, test_df, test_pos_hydro = get_data(hdf5_filename, config["label"])

	
	feature_names = train_df.drop(columns='labels').columns 

	# Prepare datasets
	center_transition, end_transition = find_transition_regions(train_df)

	if config["sampling"] !=  "None":
		## Balance training set in the transition region
		train_df = balance_dataset(train_df, center_transition, end_transition, config["sampling"])

	train_features_df = train_df.drop(columns="labels")
	train_labels = train_df["labels"]

	test_features_df = test_df.drop(columns="labels")
	test_labels = test_df["labels"]

	# keep position for 2PCF test
	test_pos = test_df[["x_dmo", "y_dmo", "z_dmo"]] 
	## Standarize features
	scaler = StandardScaler()
	scaler.fit(train_features_df)
	train_features = scaler.transform(train_features_df)
	test_features = scaler.transform(test_features_df)
	# convert pd.dataframe to np.ndarray
	train_labels = train_labels.values
	test_labels = test_labels.values

	train = {"features": train_features, "labels": train_labels}
	test = {"features": test_features, "labels": test_labels}

	if "feature_optimization" in config.keys():
		# Perform feature optimization"
		train["features"], test["features"] = feature_optimization(
			train, test, config["feature_optimization"], experiment=experiment

		)
		
		feature_names = [f'PCA_{i}' for i in range(train["features"].shape[1])]

	# -------------------------------------------------------------------------
	# Set-up and Run inference model
	# -------------------------------------------------------------------------

	trained_model = fit(train, config["model"])

	test["pred"] = prediction(trained_model, test["features"])
	
	# -------------------------------------------------------------------------
	# Save output's visualizations
	# -------------------------------------------------------------------------
	

	visualize.plot_confusion_matrix(
		test["labels"],
		test["pred"],
		classes=["Dark", "Luminous"],
		normalize=True,
		experiment=experiment,
	)

 
	# Confusion matrix for objects below the central region, where most of the training set is dark objects
	low_mass_threshold = test_features_df.M200c.values < center_transition
	test_pred_low_mass = prediction(trained_model, test['features'][low_mass_threshold])
	visualize.plot_confusion_matrix(test['labels'][low_mass_threshold], 
									test_pred_low_mass,
									classes = ['Dark', 'Luminous'],
									normalize = True, 
									experiment = experiment, 
									log_name = 'Low Mass')		

	# Confusion matrix for objects that are above the central region but bellow the mass at which all objects are luminous.
	# Here the training set is mostly luminous objects.
	high_mass_threshold = (test_features_df.M200c.values > center_transition) & (test_features_df.M200c.values < end_transition)
	test_pred_high_mass = prediction(trained_model, test['features'][high_mass_threshold])
	visualize.plot_confusion_matrix(test['labels'][high_mass_threshold], 
									test_pred_high_mass,  
									classes = ['Dark', 'Luminous'],
									normalize = True, 
									experiment = experiment, 
									log_name = 'High Mass')		

	# Feature importance

	if 'rnf' in model:
		visualize.plot_feature_importance(trained_model, 
				feature_names, 
				experiment = experiment)
	# tpcf
	label_test_positions = test_pos_hydro[test["labels"], :]
	pred_test_positions = (np.vstack([test_pos.x_dmo, test_pos.y_dmo, test_pos.z_dmo]).T)[test["pred"], :]

	visualize.plot_tpcf(
	  pred_test_positions, label_test_positions, experiment=experiment
	)



	# Metrics  
	precision = precision_score(test["labels"], test["pred"])
	metrics = {"precision": precision}

	experiment.log_metrics(metrics)
	
# TODO: change to autoconfig
if __name__ == "__main__":

	main("rnf")
