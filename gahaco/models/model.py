import os, time
from absl import flags, app
import logging
import importlib
from datetime import datetime
import pickle

import numpy as np
import pandas as pd
import comet_ml
from comet_ml import Experiment, OfflineExperiment, Optimizer

from sklearn.model_selection import train_test_split

import lightgbm
#from lightgbm import LGBMClassifier, LGBMRegressor

#import catboost
#from catboost import CatBoostClassifier, CatBoostRegressor

#from gahaco.models.train import training
from gahaco.utils.config import load_config
from gahaco.utils.optimize import merge_configs


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------
class Model():
    
    def __init__(self, FLAGS, main_config, opt_config_file_path):
        self.FLAGS = FLAGS
        self.kind = main_config["model"]["module"]

        if FLAGS.optimize_model:
            # model-/hyper-parameter optimization (run many experiments)
            self.optimize_hyper_param(main_config, opt_config_file_path)

        else:
            # run one experiment
            self.init_comet_experiment()


    def optimize_hyper_param(self, main_config, opt_config_file_path):
        """
        Optimizer model hyper-parameters and upload results to cloud.
        """
        self.opt_config_file_path = opt_config_file_path
        
        print("# load model-optimization config")
        config = load_config(
            config_file_path=self.opt_config_file_path, purpose="optimize_tree"
        )
        self.opt = Optimizer(
            config,
            api_key="VNQSdbR1pw33EkuHbUsGUSZWr",
            project_name="general",
            workspace="florpi",
            experiment_class="OfflineExperiment",
            offline_directory="/cosma/home/dp004/dc-beck3/4_GaHaCo/GaHaCo/comet/",
            #offline_directory="/cosma/home/dp004/dc-cues1/GaHaCo/comet/",
        )


    def init_comet_experiment(self):
        if self.FLAGS.upload:
            # Save experimental data in cloud
            experiment = Experiment(
                api_key="VNQSdbR1pw33EkuHbUsGUSZWr",
                project_name="general",
                workspace="florpi"
            )
        
        else:
            # Save experimental data locally
            experiment = OfflineExperiment(
                api_key="VNQSdbR1pw33EkuHbUsGUSZWr",
                project_name="general",
                workspace="florpi",
                offline_directory="/cosma/home/dp004/dc-cues1/GaHaCo/comet/",
            )
        self.experiment = experiment
    
    def fit(self, x_train, y_train, arg_model):
        """
        Train/fit the model using the training data.
        """
        if self.kind == "lightgbm":

            # Create the LightGBM training data containers
            # TODO: put parameters into config
            x, x_eval, y, y_eval = train_test_split(
                x_train,
                y_train,
                test_size=0.2,
                random_state=42,
                #stratify=y_train
            )
            lgb_train = lightgbm.Dataset(
                x,
                label=y,
                #categorical_feature=["N_subhalos", "Nmergers"], TODO Why not int?
            )
            lgb_eval = lightgbm.Dataset(
                x_eval,
                label=y_eval,
                #categorical_feature=["N_subhalos", "Nmergers"],
            )

            # Initiate RNF-horizontal-tree and create forest
            # TODO: put parameters into config
            model = lightgbm.train(
                arg_model["parameters"],
                lgb_train,
                valid_sets=lgb_eval,
                num_boost_round=50,
                early_stopping_rounds=10,
            )

            return model

        elif self.kind == "catboost":
            # for catboost label needs to be int/float
            y_train *= 1
            y_train = y_train.replace(0, -1)
            
            # checking for any categorical features
            cate_features_index = np.where(x_train.dtypes != float)[0]

            x, x_eval, y, y_eval = train_test_split(
                x_train,
                y_train,
                test_size=0.2,
                random_state=42,
                stratify=y_train
            )
            
            if arg_model["class"] == "CatBoostClassifier":
                from catboost import CatBoostClassifier
                model = CatBoostClassifier(**arg_model["parameters"])
            elif arg_model["class"] == "CatBoostRegressor":
                from catboost import CatBoostRegressor
                model = CatBoostRegressor(**arg_model["parameters"])
           
            model.fit(
                x,
                y,
                cat_features=cate_features_index,
                eval_set=(x_eval,y_eval),
            )

            return model

        else:
            model_module = importlib.import_module(arg_model["module"])
            model = getattr(model_module, arg_model['class'])

            # Initiate RNF-vertical-tree
            model_instance = model(**arg_model["parameters"])
            # Create forest
            model_instance.fit(x_train, y_train) 

            return model_instance
    
    
    def predict(self, model, test_features, arg_model=None):
        """
        Perform prediction after having trained/fitted the model.
        """
        if self.kind == "lightgbm":
            probabilities = model.predict(test_features, num_iteration=model.best_iteration)
            return  probabilities > 0.50

        elif self.kind == "catboost":
            probabilities = model.predict(test_features)
            return probabilities > 0.50

        else:
            return model.predict(test_features)


