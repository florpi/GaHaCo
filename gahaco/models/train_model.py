import os, time
<<<<<<< HEAD
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

from gahaco.utils.datautils import get_data, balance_dataset, find_transition_regions
from gahaco.utils.optimize import feature_optimization
from gahaco.utils.config import load_config
from gahaco.visualization import visualize
from gahaco.models.predict_model import prediction
from gahaco.models.fit_model import fit
from gahaco.models import hod


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
=======
from absl import flags, app
import logging
import importlib
from datetime import datetime
import pickle

import numpy as np
import pandas as pd
from comet_ml import Experiment, OfflineExperiment

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import (
	f1_score,
	confusion_matrix,
)

from gahaco.features.feature_utils import (
	get_data,
	balance_dataset,
	load_positions,
)
from gahaco.utils.optimize import feature_optimization
from gahaco.utils.config import load_config
from gahaco.utils.tpcf import compute_tpcf
from gahaco.visualization import visualize
from gahaco.models.predict_model import prediction
from gahaco.models.fit_model import fit 
from gahaco.models import hod
from gahaco.utils import feature_importance
from gahaco.features.correlation import select_uncorrelated_features



# -----------------------------------------------------------------------------
# Flags 
# -----------------------------------------------------------------------------
flags.DEFINE_string('model', 'rnf', 'model to run') # name ,default, help
flags.DEFINE_string('models_dir', '/cosma6/data/dp004/dc-cues1/tree_models/',
							'dir to save models for error analysis')
flags.DEFINE_integer('np', 1, 'Number of processes to run') 
flags.DEFINE_integer('n_splits', 4, 'Number of folds for cross-validation') 
flags.DEFINE_boolean('upload', False, 'upload model to comet ml, otherwise save in temporary folder') 
flags.DEFINE_boolean('logging', False, 'save log files') 
flags.DEFINE_boolean('mass_balance', False, 'balance dataset in different mass bins') 
FLAGS=flags.FLAGS
>>>>>>> florpi

# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------
<<<<<<< HEAD
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

    feature_names = train_df.drop(columns="labels").columns

    # Prepare datasets
    center_transition, end_transition = find_transition_regions(train_df)

    if config["sampling"] != "None":
        ## Balance training set in the transition region
        train_df = balance_dataset(
            train_df, center_transition, end_transition, config["sampling"]
        )

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

        feature_names = [f"PCA_{i}" for i in range(train["features"].shape[1])]

    # -------------------------------------------------------------------------
    # Set-up and Run inference model
    # -------------------------------------------------------------------------

    trained_model = fit(train, config["model"])

    test["pred"] = prediction(trained_model, test["features"], config["model"])

    test["pred"] = test["pred"] > 0

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

    # Confusion matrix for objects below the central region, where most of the
    # training set is dark objects
    low_mass_threshold = test_features_df.M200c.values < center_transition
    test_pred_low_mass = prediction(
        trained_model, test["features"][low_mass_threshold], config["model"]
    )
    visualize.plot_confusion_matrix(
        test["labels"][low_mass_threshold],
        test_pred_low_mass,
        classes=["Dark", "Luminous"],
        normalize=True,
        experiment=experiment,
        log_name="Low Mass",
    )

    # Confusion matrix for objects that are above the central region but bellow
    # the mass at which all objects are luminous.
    # Here the training set is mostly luminous objects.
    high_mass_threshold = (test_features_df.M200c.values > center_transition) & (
        test_features_df.M200c.values < end_transition
    )
    test_pred_high_mass = prediction(
        trained_model, test["features"][high_mass_threshold], config["model"]
    )
    visualize.plot_confusion_matrix(
        test["labels"][high_mass_threshold],
        test_pred_high_mass,
        classes=["Dark", "Luminous"],
        normalize=True,
        experiment=experiment,
        log_name="High Mass",
    )

    # Feature importance

    if "rnf" in model:
        visualize.plot_feature_importance(
            trained_model, feature_names, experiment=experiment
        )

    # tpcf
    halo_occ = hod.HOD(train_df)
    halo_occ.df = test_df
    n_hod_galaxies = halo_occ.populate()
    n_hod_galaxies = n_hod_galaxies > 0
    visualize.plot_confusion_matrix(
        test["labels"],
        n_hod_galaxies,
        classes=["Dark", "Luminous"],
        normalize=True,
        experiment=experiment,
        log_name="HOD",
    )

    hod_positions = (np.vstack([test_pos.x_dmo, test_pos.y_dmo, test_pos.z_dmo]).T)[
        n_hod_galaxies, :
    ]
    label_test_positions = test_pos_hydro[test["labels"], :]
    pred_test_positions = (
        np.vstack([test_pos.x_dmo, test_pos.y_dmo, test_pos.z_dmo]).T
    )[test["pred"], :]

    visualize.plot_tpcf(
        pred_test_positions, label_test_positions, hod_positions, experiment=experiment
    )

    # Metrics
    precision = precision_score(test["labels"], test["pred"])
    metrics = {"precision": precision}

    experiment.log_metrics(metrics)


# TODO: change to autoconfig
if __name__ == "__main__":

    main("rnf_pca")
    # main('lightgbm')
=======
def main(argv):
	# -----------------------------------------------------------------------------
	# Loggings
	# -----------------------------------------------------------------------------

	if FLAGS.upload:
		experiment = Experiment(
			api_key="VNQSdbR1pw33EkuHbUsGUSZWr", project_name="general", workspace="florpi"
		)
	else:
		print('Correctly going offline')
		experiment = OfflineExperiment(
			api_key="VNQSdbR1pw33EkuHbUsGUSZWr", project_name="general", workspace="florpi",
			offline_directory="/tmp/"
		)

	if FLAGS.logging:
		tag_datetime = datetime.now().strftime("%H%M_%d%m%Y")
		logging.basicConfig(
			filename="../../models/rnf/log_%s.log" % tag_datetime,
			level=logging.INFO,
			format="%(asctime)s - %(levelname)s - %(message)s",
			datefmt="%d-%b-%y %H:%M:%S",
		)
		logging.info("")
		logging.info("GROWING TREES")
		logging.info("")

	config_file_path = "../../models/%s/config_%s.json" % (FLAGS.model, FLAGS.model)
	config = load_config(config_file_path=config_file_path)

	config['model']['parameters']['n_jobs'] = FLAGS.np
	print(f'Using {FLAGS.np} cores to fit models')
	print(config['model']['parameters'])
	# -------------------------------------------------------------------------
	# Load and prepare datasets
	# -------------------------------------------------------------------------

	# Load dataset
	features, labels = get_data(config["label"])
	# Select feautures by correlation
	#features = select_uncorrelated_features(features)
	# K-Fold validation
	metric_module = importlib.import_module(config["metric"]["module"])
	metric = getattr(metric_module, config["metric"]["method"])

	if "sampling" in config:
		sampler_module = importlib.import_module(config["sampling"]["module"])
		sampler = getattr(sampler_module, config["sampling"]["method"])

	importances, sk_importances, coefficients,cms,hod_cms,hydro_tpcf,pred_tpcf,hod_tpcf=([] for i in range(8))

	if config['label']=='stellar_mass':
		skf =KFold(n_splits=FLAGS.n_splits, shuffle=True)
	else:
		skf = StratifiedKFold(n_splits=FLAGS.n_splits, shuffle=True)
	fold=0
	for train_idx, test_idx in skf.split(features, labels):
		x_train, x_test = (features.iloc[train_idx], features.iloc[test_idx])
		y_train, y_test = (labels.iloc[train_idx], labels.iloc[test_idx])

		halo_occ = hod.HOD(x_train, y_train)
		halo_occ.df = x_test
		n_hod_galaxies=halo_occ.populate()
		n_hod_galaxies = n_hod_galaxies > 0

		if "sampling" in config:
			if FLAGS.mass_balance:
				x_train, y_train = balance_dataset(x_train, y_train,
					sampler)
			else:
				x_train, y_train = balance_dataset(x_train, y_train,
					sampler, split=None)

		## Standarize features
		scaler = StandardScaler()
		scaler.fit(x_train)
		x_train_scaled = scaler.transform(x_train)
		x_train = pd.DataFrame(x_train_scaled, index=x_train.index, columns=x_train.columns)
		x_test_scaled = scaler.transform(x_test)
		x_test = pd.DataFrame(x_test_scaled, index=x_test.index, columns=x_test.columns)

		# -------------------------------------------------------------------------
		# Set-up and Run inference model
		# -------------------------------------------------------------------------
		trained_model = fit(x_train, y_train, config["model"])

		if FLAGS.models_dir:
			with open(FLAGS.models_dir+f'{FLAGS.model}_{fold}','wb') as f:
				    pickle.dump(trained_model, f)
			x_test.to_hdf(FLAGS.models_dir+f'{FLAGS.model}_{fold}_test_feats.hdf5', key='df', mode='w')
			y_test.to_hdf(FLAGS.models_dir+f'{FLAGS.model}_{fold}_test_labels.hdf5', key='df', mode='w')

		y_pred = prediction(trained_model, x_test, config["model"])
		
		metric_value = metric(y_test, y_pred, **config["metric"]["params"])
		experiment.log_metric("Metric value", metric_value)

		if config['label']=='stellar_mass':
			visualize.regression(y_test, y_pred, metric_value, fold=fold, experiment=experiment)
			y_pred = y_pred > config['log_stellar_mass_threshold']
			y_test = y_test > config['log_stellar_mass_threshold']
		cm = confusion_matrix(y_test, y_pred)
		cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
		cms.append(cm)

		cm = confusion_matrix(y_test, n_hod_galaxies)
		cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
		hod_cms.append(cm)
		imp = feature_importance.dropcol(trained_model, x_train, y_train,
					x_test, y_test, metric_value, metric, config['metric']['params'])
		importances.append(imp)
		sk_importances.append(trained_model.feature_importances_)

		test_hydro_pos, test_dmo_pos= load_positions(test_idx)
		r_c, test_hydro_tpcf = compute_tpcf(test_hydro_pos[y_test>0])
		hydro_tpcf.append(test_hydro_tpcf)
		pred_tpcf.append(compute_tpcf(test_dmo_pos[y_pred>0])[1])
		hod_tpcf.append(compute_tpcf(test_dmo_pos[n_hod_galaxies])[1])
		fold+=1

	# -------------------------------------------------------------------------
	# Save output's visualizations
	# -------------------------------------------------------------------------

	print(cms)

	visualize.plot_confusion_matrix(cms,
									classes = ['Dark', 'Luminous'],
									normalize = False,
									title='Tree',
									experiment = experiment,
									)
	visualize.plot_confusion_matrix(hod_cms,
									classes = ['Dark', 'Luminous'],
									normalize = False,
									title='HOD',
									experiment = experiment,
									)

	visualize.plot_feature_importance(importances,
								x_train.columns,
								title='Drop column',
								experiment=experiment)

	visualize.plot_feature_importance(sk_importances,
								x_train.columns,
								title='Gini impurity',
								experiment=experiment)

	visualize.plot_tpcfs(
					r_c, hydro_tpcf, pred_tpcf, hod_tpcf, experiment=experiment
						)

	experiment.add_tag(f'preprocessing = {config["sampling"]}')
	experiment.add_tag(f'classifier = {FLAGS.model}')

	print('All good :)')
	
if __name__ == "__main__":
	app.run(main)
>>>>>>> florpi
