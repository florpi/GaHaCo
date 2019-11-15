import os, time
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

# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------
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
