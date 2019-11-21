import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm
import importlib

# import xgboost


def fit(x_train, y_train, arg_model):
	"""
	"""
	if arg_model["module"] == "lightgbm":

		# Create the LightGBM training data containers
		# TODO: put parameters into config
		x, x_eval, y, y_eval = train_test_split(
			x_train,
			y_train,
			test_size=0.2,
			random_state=42,
			stratify=y_train
		)
		lgb_train = lightgbm.Dataset(x, label=y)
		lgb_eval = lightgbm.Dataset(x_eval, label=y_eval)

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

	elif arg_model["module"] == "catboost":
        #TODO
        pass

	else:

		model_module = importlib.import_module(arg_model["module"])
		model = getattr(model_module, arg_model['class'])

		# Initiate RNF-vertical-tree
		model_instance = model(**arg_model["parameters"])
		# Create forest
		model_instance.fit(x_train, y_train) 

		return model_instance
		
