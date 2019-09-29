import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import lightgbm

# import xgboost


def prediction(train, test, arg_model):

    if arg_model["module"] == "sklearn.ensemble":

        if arg_model["class"] == "RandomForestClassifier":
            rf = RandomForestClassifier(**arg_model["parameters"])
            rf.fit(train["features"], train["labels"])
            return rf.predict(test_features_std)

    if arg_model["module"] == "lightgbm":

        # Create the LightGBM training data containers
        # TODO: put parameters into config
        x, x_test, y, y_test = train_test_split(
            train["features"],
            train["labels"],
            test_size=0.2,
            random_state=42,
            stratify=train["features"],
        )

        lgb_train = lightgbm.Dataset(
            x,
            label=y,
            # categorical_feature=list(feature_names)
        )
        lgb_eval = lightgbm.Dataset(x_test, label=y_test)

        # TODO: put parameters into config
        model = lightgbm.train(
            arg_model["parameters"],
            lgb_train,
            valid_sets=lgb_eval,
            num_boost_round=50,
            early_stopping_rounds=10,
        )

        return model.predict(test["features"])
