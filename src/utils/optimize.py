import numpy as np
from typing import Any, Callable

from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


import pandas as pd


def feature_optimization(train, test, arg, experiment=None):
    """
    Perform feature decompositions and importance measures to create a optimized
    set of features with low feature variance or correlation.
    """

    if arg["primary"] == "PCA":
        train_features, test_features = pca_transform(
            train["features"],
            test["features"],
            arg["secondary"],
            experiment,
        )
        return train_features, test_features

    elif arg["primary"] == "LDA":
        train_features, test_features = lda_transform(
            train["features"],
            train["labels"],
            test["features"],
            test["labels"],
            arg["secondary"],
            experiment,
        )
        return train_features, test_features


def pca_transform(train, test, arg_pca, experiment=None):
    """
    """

    if isinstance(arg_pca, (dict)):
        return _pca_dict(train, test, arg_pca)

    elif isinstance(arg_pca, (float)):
        return _pca_corrlimit(train, test, arg_pca)

    elif arg_pca == "cross_val":
        return _pca_cross_val(train, test)


def _pca_dict(train, test, arg_pca: dict):
    """
    """
    pca = PCA(**arg_pca)

    # Perform feature optimization
    train = pca.fit_transform(train)
    test = pca.transform(test)

    return train, test


def _pca_corrlimit(train, test, correlation_limit: float):
    """
    """
    # convert np.ndarray to pd.dataframe
    if isinstance(train_features_std, (np.ndarray)):
        df_train = pd.DataFrame(
            data=train,
            index=np.arange(train.shape[0]),
            columns=["orig_%d" % ff for ff in range(train.shape[1])],
        )

        df_test = pd.DataFrame(
            data=test,
            index=np.arange(test.shape[0]),
            columns=["orig_%d" % ff for ff in range(test.shape[1])],
        )

    # Create correlation matrix
    corr_matrix = df_train.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    # Find index of feature columns with correlation greater than correlation_limit
    to_drop = [column for column in upper.columns if any(upper[column] > correlation_limit)]
    # remove highly correlated features from dataset
    df_train = df_train.drop(df_train[to_drop], axis=1)
    df_test = df_test.drop(df_test[to_drop], axis=1)

    pca = PCA(n_components=len(df_train.columns.values))

    # Perform feature optimization
    train = pca.fit_transform(df_train.values)
    test = pca.transform(df_test.values)

    return train, test


def _pca_cross_val(train, test):
    """
    """
    pca = PCA(svd_solver='full')

    pca_scores = []
    n_components = np.arange(0, train.shape[1], 1)

    # determine cross-val. score for different dimensions of feature space
    for n in n_components:
        pca.n_components = n
        pca_scores.append(np.mean(cross_val_score(pca, train, cv=5)))

    # choose # of dimensions with highest cross-val. score
    n_components = n_components[np.argmax(pca_scores)]
    pca.n_components = n_components

    # Perform feature optimization
    train = pca.fit_transform(train)
    test = pca.transform(test)

    return train, test


def lda_transform(
    train_features, train_labels,
    test_features, test_labels,
    arg_lda):
    """
    """

    if isinstance(arg_lda, (float)):
        return _lda_varlimit(
            train_features, train_labels,
            test_features, test_labels,
            arg_lda,
        )
    elif isinstance(arg_lda, (dict)):
        return _lda_dict(
            train_features, train_labels,
            test_features, test_labels,
            arg_lda,
        )


def _lda_dict(
    train_features, train_labels,
    test_features, test_labels,
    arg_lda: dict):
    """
    """
    # Create LDA
    lda = LDA(**arg_lda)

    # tranform features based on training dataset
    train_features = lda.fit_transform(train_features, train_labels)
    test_features = lda.transform(test_features)

    print("explained_variance_ratio_ ", lda.explained_variance_ratio_)

    return train_features, test_features


def _lda_varlimit(
    train_features, train_labels,
    test_features, test_labels,
    variance_limit: float):
    """
    """
    # Create array of explained variance ratios
    lda = LDA(n_components=None)
    print(train_features.shape)
    dump = lda.fit(train_features, train_labels)
    lda_var_ratios = lda.explained_variance_ratio_
    print("explained_variance_ratio_", lda_var_ratios)

    # Set initial variance
    total_variance = 0.0
    # Set initial number of features
    n_components = 0

    # Run through explained variance of each feature:
    for explained_variance in lda_var_ratios:

        # Add the explained variance to the total
        total_variance += explained_variance

        # Add one to the number of components
        n_components += 1

        if total_variance >= variance_limit:
            break

    # Create LDA
    lda = LDA(n_components=n_components, priors=None)

    # tranform features based on training dataset
    train_features = lda.fit_transform(train_features, train_labels)
    test_features = lda.transform(test_features)

    # check mean accuracy
    #score_lda = lda.score(test_features, test_labels)

    return train_features, test_features
