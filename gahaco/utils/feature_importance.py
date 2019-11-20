from sklearn.metrics import f1_score
from sklearn.base import clone
import pandas as pd
import numpy as np

def test_set_metric(model, X_test, y_test, metric, metric_params):
    # f1 socre?
    y_pred = model.predict(X_test)
    return metric(y_test, y_pred, **metric_params)

def permutation(model, X_test, y_test,
        baseline_metric,
        metric,
        metric_params,
        ):
    imp = []
    for col in X_test.columns:
        save = X_test[col].copy()
        X_test[col] = np.random.permutation(X_test[col])
        permuted_column = test_set_metric(model, X_test, y_test, metric, metric_params)
        X_test[col] = save
        imp.append(baseline_metric - permuted_column)
    return np.array(imp)

def dropcol(model, X_train, y_train, 
            X_test, y_test, 
            baseline_metric,
            metric,
            metric_params,
            ):
    imp = []
    for col in X_train.columns:
        X = X_train.drop(col, axis=1)
        X_ = X_test.drop(col, axis=1)
        model_ = clone(model)
        #model_.random_state = 999
        model_.fit(X, y_train)
        drop_column = test_set_metric(model_, X_, y_test, metric, metric_params)
        imp.append(baseline_metric - drop_column)
    imp = np.array(imp)
    return imp
    
