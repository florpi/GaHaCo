import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import lightgbm
import catboost
# import xgboost


def prediction(model, test_features, arg_model=None):
	"""
	"""

	if arg_model["module"] == "lightgbm":
		probabilities = model.predict(test_features, num_iteration=model.best_iteration)
		return  probabilities > 0.50
    elif arg_model["module"] == "catboost":
        #TODO
        pass
	else:
		return model.predict(test_features)


