from sklearn.metrics import f1_score, mean_squared_error
from gahaco.utils import summary 
from sklearn.base import clone
import pandas as pd
import numpy as np
from scipy.stats import chisquare

def test_set_metric(model, X_test, y_test, metric, metric_params):
    y_pred = model.predict(X_test)
    return metric(y_test, y_pred, **metric_params)

def permutation(model, X_test, y_test,
        baseline_metric,
        metric,
        metric_params,
        inverse=True,
        ):
    imp = []
    for col in X_test.columns:
        save = X_test[col].copy()
        X_test[col] = np.random.permutation(X_test[col])
        permuted_column = test_set_metric(model, X_test, y_test, metric, metric_params)
        X_test[col] = save
        imp.append((baseline_metric - permuted_column)/baseline_metric)
        
    imp = np.array(imp)
    if inverse:
        imp *= -1
    return imp

def dropcol(model, X_train, y_train, 
            X_test, y_test, dmo_pos_test,
            r_c, hydro_tpcf_test,
            baseline_metric,
            metric,
            metric_params,
            stellar_mass_thresholds,
            inverse=True,
            boxsize=300.
            ):
    imp, xi2 = [], []
    for col in X_train.columns:
        X = X_train.drop(col, axis=1)
        X_ = X_test.drop(col, axis=1)
        model_ = clone(model)
        #model_.random_state = 999
        model_.fit(X, y_train)
        drop_column = test_set_metric(model_, X_, y_test, metric, metric_params)
        imp.append((baseline_metric - drop_column)/baseline_metric)
        #imp.append(drop_column)
        y_pred = model_.predict(X_)
        _, model_tpcf = summary.model_stellar_mass_summary(y_test, y_pred, 
                                                                stellar_mass_thresholds,
                                                                dmo_pos_test, boxsize)
        mse = [] 
        for i in range(len(hydro_tpcf_test)):
            print(mean_squared_error(hydro_tpcf_test[i], model_tpcf[i]))
            mse.append(mean_squared_error(r_c**2*hydro_tpcf_test[i], r_c**2*model_tpcf[i]))
        xi2.append(mse)
    imp = np.array(imp)
    xi2 = np.array(xi2)
    if inverse:
        imp *= -1
    return imp, xi2
    
