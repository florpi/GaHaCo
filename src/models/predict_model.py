import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import lightgbm

# import xgboost


def prediction(train, test, arg_model):
    """
    """

    if arg_model["module"] == "sklearn.ensemble":

        if arg_model["class"] == "RandomForestClassifier":
            # Initiate RNF-vertical-tree
            rf = RandomForestClassifier(**arg_model["parameters"])
            # Create forest
            rf.fit(train["features"], train["labels"])
            # make prediction
            return rf.predict(test["features"])

    if arg_model["module"] == "lightgbm":

        # Create the LightGBM training data containers
        # TODO: put parameters into config
        x, x_eval, y, y_eval = train_test_split(
            train["features"],
            train["labels"],
            test_size=0.2,
            random_state=42,
            stratify=train["features"],
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
        
        # make prediction
        return model.predict(test["features"])
