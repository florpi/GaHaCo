import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import lightgbm

# import xgboost


def prediction(model, test_features):
	"""
	"""

	return model.predict(test_features)


