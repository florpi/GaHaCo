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
    r2_score,
    confusion_matrix,
)

from gahaco.features.feature_utils import (
    get_data,
    balance_dataset,
    load_positions,
)
from gahaco.visualization import visualize
from gahaco.utils import summary 
from gahaco.models import hod
from gahaco.models.model import Model
from gahaco.utils.optimize import merge_configs
from gahaco.utils import feature_importance
from gahaco.utils.optimize import feature_optimization
from gahaco.utils.config import load_config
from gahaco.utils.tpcf import compute_tpcf
from gahaco.features.correlation import select_uncorrelated_features

# -----------------------------------------------------------------------------
# Flags 
# -----------------------------------------------------------------------------
flags.DEFINE_string('model', 'lightgbm_reg', 'model to run') # name ,default, help
flags.DEFINE_integer('np', 2, 'Number of processes to run') 
flags.DEFINE_integer('n_splits', 4, 'Number of folds for cross-validation') 
flags.DEFINE_boolean('upload', True, 'upload model to comet.ml, otherwise save in temporary folder') 
flags.DEFINE_boolean('optimize_model', False, 'use comet.ml to perform hyper-param. optimization.') 
flags.DEFINE_boolean('logging', False, 'save log files') 
flags.DEFINE_boolean('mass_balance', False, 'balance dataset in different mass bins') 
flags.DEFINE_boolean('figures', True, 'if final figures should be created') 
FLAGS = flags.FLAGS

def main(argv):
    """
    """
    opt_config_file_path = "../../models/%s/config_optimize.json" % (FLAGS.model)
    main_config_file_path = "../../models/%s/config_%s.json" % (FLAGS.model, FLAGS.model)
    config = load_config(config_file_path=main_config_file_path, purpose="")
    print(f"Using {FLAGS.np} cores to fit models")

    # Initiate Model/Experiment
    model = Model(FLAGS, config, opt_config_file_path)

    # Load dataset
    features, labels = get_data(config["label"])
    m200c = features.M200_DMO.values
    
    # Set metric
    metric_module = importlib.import_module(config["metric"]["module"])
    metric = getattr(metric_module, config["metric"]["method"])

    # Define sampling
    if "sampling" in config:
        sampler_module = importlib.import_module(config["sampling"]["module"])
        sampler = getattr(sampler_module, config["sampling"]["method"])
    else:
        sampler = None

    # K-fold validation setting
    if config['label']=='stellar_mass':
        skf = KFold(n_splits=FLAGS.n_splits, shuffle=True)
    else:
        skf = StratifiedKFold(n_splits=FLAGS.n_splits, shuffle=True)
    
    if FLAGS.optimize_model:
        # model-/hyper-parameter optimization (run many experiments)
        for experiment in model.opt.get_experiments():
            experiment.add_tag('hyper-parameter optimization %s' % FLAGS.model)
            config = merge_configs(config, model.opt, experiment)
            train(
                model, experiment, features, labels, m200c, metric, sampler, skf, config, FLAGS
            )

    else:
        # run one experiment
        train(model, model.experiment, features, labels, m200c, metric, sampler, skf, config, FLAGS)


def train(model, experiment, features, labels, m200c, metric, sampler, skf, config, FLAGS):
    """
    """

    if ("feature_optimization" in config.keys()) and (FLAGS.optimize_model is False):
        if 'PCA' in config['feature_optimization']: 
            # TODO: Needs to be updated to only take features and return dataframe
            train_features, test_features = feature_optimization(
                train, test, config["feature_optimization"], experiment=experiment
            )

            feature_names = [f"PCA_{i}" for i in range(train["features"].shape[1])]
        elif config['feature_optimization']['uncorrelated']:
            gini_importances = np.loadtxt(f'../../models/{FLAGS.model}/gini_importances.csv')
            features = select_uncorrelated_features(features, 
                                                    gini_impurities=gini_importances,
                                                    experiment=experiment)

    dropcol_importance,pm_importance,gini_importance,cms = ([] for i in range(4))

    hod_cms,hydro_tpcf,pred_tpcf,hod_tpcfs = ([] for i in range(4))
    halo_occs = []

    fold = 0
    for train_idx, test_idx in skf.split(features, labels):
        x_train, x_test = (features.iloc[train_idx], features.iloc[test_idx])
        y_train, y_test = (labels.iloc[train_idx], labels.iloc[test_idx])


        # -----------------------------------------------------------------------------
        # BASELINE HOD MODEL EVALUATION 
        # -----------------------------------------------------------------------------

        hydro_pos_test, dmo_pos_test= load_positions(test_idx)

        if (config['label']=='stellar_mass'):
            stellar_mass_thresholds = [9, 9.2, 9.3]
            halo_occ, hod_cm, hod_tpcf = summary.hod_stellar_mass_summary(
                m200c[train_idx],
                m200c[test_idx],
                y_train,
                y_test,
                stellar_mass_thresholds,
                dmo_pos_test
            )

            r_c, hydro_tpcf_test = summary.hydro_stellar_mass_summary(
                hydro_pos_test,
                y_test,
                stellar_mass_thresholds
            )

        else:
            stellar_mass_thresholds = [9]
            halo_occ, hod_cm, hod_tpcf = summary.hod_summary(m200c[train_idx], m200c[test_idx], 
                                                       y_train, y_test, dmo_pos_test)

            r_c, hydro_tpcf_test = summary.hydro_summary(hydro_pos_test, y_test)

        hydro_tpcf.append(hydro_tpcf_test)

        halo_occs.append(halo_occ)
        hod_cms.append(hod_cm)
        hod_tpcfs.append(hod_tpcf)

        # -----------------------------------------------------------------------------
        # PREPROCESS DATASET FOR TRAINING (balancing + normalisation)
        # -----------------------------------------------------------------------------


        if sampler is not None:
            if FLAGS.mass_balance:
                    x_train, y_train = balance_dataset(x_train, y_train,
                        sampler)
            else:
                x_train, y_train = balance_dataset(
                    x_train, y_train, sampler, split=None
                )

        ## Standarize features
        scaler = StandardScaler()
        scaler.fit(x_train)
        x_train_scaled = scaler.transform(x_train)
        x_train = pd.DataFrame(
            x_train_scaled,
            index=x_train.index,
            columns=x_train.columns
        )
        x_test_scaled = scaler.transform(x_test)
        x_test = pd.DataFrame(
            x_test_scaled,
            index=x_test.index,
            columns=x_test.columns
        )

        # -----------------------------------------------------------------------------
        # FIT MODEL
        # -----------------------------------------------------------------------------

        trained_model = model.fit(x_train, y_train, config["model"])
        y_pred = model.predict(trained_model, x_test, config["model"])

        metric_value = metric(y_test, y_pred, **config["metric"]["params"])
        experiment.log_metric("mean_squared_error", metric_value)


        # -----------------------------------------------------------------------------
        # SAVE FEATURE IMPORTANCE AND EVALUATION METRIC
        # -----------------------------------------------------------------------------

        if (config['label']=='stellar_mass') or (config['label']=='nr_of_satellites'):
            threshold = (y_test > 0.) & (y_pred > 0.)
            r2 = r2_score(y_test[threshold], y_pred[threshold])
            visualize.regression(
                y_test[threshold], y_pred[threshold], r2, metric_value, fold=fold, experiment=experiment
            )
        if FLAGS.optimize_model is False:

            if (config['label']=='stellar_mass') or (config['label']=='nr_of_satellites'):
                threshold = (y_test > 0.) & (y_pred > 0.)
                r2 = r2_score(y_test[threshold], y_pred[threshold])
                visualize.regression(
                    y_test[threshold],
                    y_pred[threshold],
                    r2,
                    metric_value,
                    fold=fold,
                    experiment=experiment
                )
                visualize.histogram(
                        y_test[threshold], y_pred[threshold], experiment
                )

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

            if (config['label']=='stellar_mass'):

                cm, model_tpcf = summary.model_stellar_mass_summary(y_test, y_pred, 
                                                                stellar_mass_thresholds,
                                                                dmo_pos_test)
            else:
                cm, model_tpcf = summary.model_summary(y_test, y_pred, dmo_pos_test)

            cms.append(cm)
            pred_tpcf.append(model_tpcf)
            fold+=1

    # -----------------------------------------------------------------------------
    # SUMMARY FIGURES
    # -----------------------------------------------------------------------------

    if (FLAGS.optimize_model is False) and (FLAGS.figures is True):
        if (config['label'] != 'nr_of_satellites'):
            # ---------------------------------------------------------------------
            # Save output's visualizations
            # ---------------------------------------------------------------------
            visualize.plot_confusion_matrix(
                cms,
                classes = ['Dark', 'Luminous'],
                normalize = False,
                title='LGBM',
                experiment = experiment,
                stellar_mass_thresholds=stellar_mass_thresholds
            )
            visualize.plot_confusion_matrix(
                hod_cms,
                classes = ['Dark', 'Luminous'],
                normalize = False,
                title='HOD',
                experiment = experiment,
                stellar_mass_thresholds=stellar_mass_thresholds
            )
            visualize.plot_tpcfs(
                r_c, hydro_tpcf, pred_tpcf, hod_tpcfs, experiment=experiment,
                stellar_mass_thresholds=stellar_mass_thresholds
            )

            visualize.plot_tpcfs(
                r_c, hydro_tpcf, None, hod_tpcfs, experiment=experiment,
                stellar_mass_thresholds=stellar_mass_thresholds
            )

        else:
            visualize.mean_halo_occupation(halo_occs, experiment=experiment)

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
            #visualize.plot_tpcfs(
            #    r_c, hydro_tpcf, pred_tpcf, hod_tpcf, experiment=experiment
            #)


            if config['feature_optimization']['save_importance']: 
                np.savetxt(
                    f'../../models/{FLAGS.model}/gini_importances.csv',
                    np.mean(gini_importance, axis=0)
                )
        experiment.add_tag(f'classifier = {FLAGS.model}')

    print('All good :)')

if __name__ == "__main__":
    app.run(main)
