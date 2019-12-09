import os, time
import logging
from absl import flags, app
import importlib
from datetime import datetime
import pickle

import numpy as np
import pandas as pd
from comet_ml import Experiment, OfflineExperiment, Optimizer

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
from gahaco.visualization import visualize
from gahaco.models.predict import prediction
from gahaco.models.fit import fitting
from gahaco.models import hod
from gahaco.models.model import Model
from gahaco.utils.optimize import merge_configs
from gahaco.utils import feature_importance
from gahaco.utils.optimize import feature_optimization
from gahaco.utils.config import load_config
#from gahaco.utils.tpcf import compute_tpcf
from gahaco.features.correlation import select_uncorrelated_features

# -----------------------------------------------------------------------------
# Flags 
# -----------------------------------------------------------------------------
flags.DEFINE_string('model', 'rnf', 'model to run') # name ,default, help
flags.DEFINE_integer('np', 10, 'Number of processes to run') 
flags.DEFINE_integer('n_splits', 1, 'Number of folds for cross-validation') 
flags.DEFINE_boolean('upload', True, 'upload model to comet.ml, otherwise save in temporary folder') 
flags.DEFINE_boolean('optimize_model', True, 'use comet.ml to perform hyper-param. optimization.') 
flags.DEFINE_boolean('logging', False, 'save log files') 
flags.DEFINE_boolean('mass_balance', False, 'balance dataset in different mass bins') 
flags.DEFINE_boolean('figures', False, 'if final figures should be created') 
FLAGS = flags.FLAGS

def main(argv):
    """
    """
    opt_config_file_path = "../../models/%s/config_optimize.json" % (FLAGS.model)
    main_config_file_path = "../../models/%s/config_%s.json" % (FLAGS.model, FLAGS.model)
    config = load_config(config_file_path=main_config_file_path, purpose="")
    config['model']['parameters']['n_jobs'] = FLAGS.np
    print(f"Using {FLAGS.np} cores to fit models")

    # Initiate Model/Experiment
    model = Model(FLAGS, config, opt_config_file_path)

    # Load dataset
    features, labels = get_data(config["label"])
    m200c = features.M200c.values
    
    # Set metric
    metric_module = importlib.import_module(config["metric"]["module"])
    metric = getattr(metric_module, config["metric"]["method"])

    # Define sampling
    if "sampling" in config:
        sampler_module = importlib.import_module(config["sampling"]["module"])
        sampler = getattr(sampler_module, config["sampling"]["method"])

    # K-fold validation setting
    if config['label']=='stellar_mass':
        skf = KFold(n_splits=FLAGS.n_splits, shuffle=True)
    else:
        skf = StratifiedKFold(n_splits=FLAGS.n_splits, shuffle=True)
    
    if FLAGS.optimize_model:
        # model-/hyper-parameter optimization (run many experiments)
        for experiment in model.opt.get_experiments():
            experiment.add_tag('hyper-parameter optimization 1')
            config = merge_configs(config, model.opt, experiment)
            train(
                model, experiment, features, labels, m200c, metric, sampler, skf, config, FLAGS)

    else:
        # run one experiment
        train(model, model.experiment, features, labels, m200c, metric, sampler, skf, config, FLAGS)


def train(model, experiment, features, labels, m200c, metric, sampler, skf, config, FLAGS):
    """
    """

    if ("feature_optimization" in config.keys()) and (FLAGS.optimize_model is False):
        if config['feature_optimization']['PCA']: 
            # TODO: Needs to be updated to only take features and return dataframe
            train_features, test_features = feature_optimization(
                train, test, config["feature_optimization"], experiment=experiment
            )

            feature_names = [f"PCA_{i}" for i in range(train["features"].shape[1])]
        elif config['feature_optimization']['uncorrelated']:
            gini_importances = np.loadtxt(f'../../models/{FLAGS.model}/gini_importances.csv')
            features = select_uncorrelated_features(features, gini_importances)

    dropcol_importance,pm_importance,gini_importance,cms = ([] for i in range(4))
    hod_cms,hydro_tpcf,pred_tpcf,hod_tpcf = ([] for i in range(4))

    fold=0
    for train_idx, test_idx in skf.split(features, labels):
        x_train, x_test = (features.iloc[train_idx], features.iloc[test_idx])
        y_train, y_test = (labels.iloc[train_idx], labels.iloc[test_idx])

        halo_occ = hod.HOD(m200c[train_idx], y_train)
        halo_occ.m200c = m200c[test_idx] 
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

        # Set-up and Run inference model
        trained_model = model.fit(x_train, y_train, config["model"])
        y_pred = model.predict(trained_model, x_test, config["model"])


        metric_value = metric(y_test, y_pred, **config["metric"]["params"])
        experiment.log_metric("Metric value", metric_value)

        if config['label']=='stellar_mass':
            visualize.regression(
                y_test, y_pred, metric_value, fold=fold, experiment=experiment
            )
            y_pred = y_pred > config['log_stellar_mass_threshold']
            y_test = y_test > config['log_stellar_mass_threshold']
        cm = confusion_matrix(y_test, y_pred)
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        
        
        experiment.log_image(cm, name="Confusion Matrix")
        cms.append(cm)

        if FLAGS.optimize_model is False:
            cm = confusion_matrix(y_test, n_hod_galaxies)
            cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
            hod_cms.append(cm)

            if config['feature_optimization']['measure_importance']:
                imp = feature_importance.dropcol(
                    trained_model,
                    x_train,
                    y_train,
                    x_test,
                    y_test,
                    metric_value,
                    metric,
                    config['metric']['params']
                )
                dropcol_importance.append(imp)
                imp = feature_importance.permutation(
                    trained_model, 
                    x_test,
                    y_test,
                    metric_value,
                    metric,
                    config['metric']['params']
                )
                pm_importance.append(imp)

                gini_importance.append(trained_model.feature_importances_)

            test_hydro_pos, test_dmo_pos= load_positions(test_idx)
            r_c, test_hydro_tpcf = compute_tpcf(test_hydro_pos[y_test>0])
            hydro_tpcf.append(test_hydro_tpcf)
            pred_tpcf.append(compute_tpcf(test_dmo_pos[y_pred>0])[1])
            hod_tpcf.append(compute_tpcf(test_dmo_pos[n_hod_galaxies])[1])
            fold+=1

    if (FLAGS.optimize_model is False) and (FLAGS.figures is True):
        # ---------------------------------------------------------------------
        # Save output's visualizations
        # ---------------------------------------------------------------------
        visualize.plot_confusion_matrix(
            cms,
            classes = ['Dark', 'Luminous'],
            normalize = False,
            title='Tree',
            experiment = experiment,
        )
        visualize.plot_confusion_matrix(
            hod_cms,
            classes = ['Dark', 'Luminous'],
            normalize = False,
            title='HOD',
            experiment = experiment,
        )
        visualize.plot_tpcfs(
            r_c, hydro_tpcf, pred_tpcf, hod_tpcf, experiment=experiment
        )

        if config['feature_optimization']['measure_importance']:
            visualize.plot_feature_importance(
                dropcol_importance,
                x_train.columns,
                title='Drop column',
                experiment=experiment
            )
            visualize.plot_feature_importance(
                pm_importance,
                x_train.columns,
                title='Permute column',
                experiment=experiment
            )
            visualize.plot_feature_importance(
                gini_importance,
                x_train.columns,
                title='Gini impurity',
                experiment=experiment,
            )

            if not config['feature_optimization']['uncorrelated']:
                np.savetxt(
                    f'../../models/{FLAGS.model}/gini_importances.csv',
                    np.mean(gini_importance, axis=0)
                )
        sampling_method = config['sampling']['method']
        experiment.add_tag(f'sampling = {sampling_method}')
        experiment.add_tag(f'classifier = {FLAGS.model}')

    print('All good :)')

if __name__ == "__main__":
    app.run(main)