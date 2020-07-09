import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
from collections import defaultdict
from gahaco.visualization.visualize import rename_features

def select_uncorrelated_features(features_df, labels,
                                gini_impurities=None,
                                method='average', distance_cutoff=0.3,
                                experiment=None):
    '''
    Clusters the Spearman rank-roder correlation of the different features, we keep only a single feature per cluster,  
    if gini impurities are given keeps the one that was most important for the classificiation at hand.

    Args:
        features_df: pandas data frame of features
    Returns:
        reduced dataframe with only low correlated features
    '''
    corr = np.round(spearmanr(features_df).correlation, 4)
    corr_condensed = hierarchy.distance.squareform(1-abs(corr))
    print("hello :-)")
    corr_linkage = hierarchy.linkage(corr_condensed, method=method)
    cluster_ids = hierarchy.fcluster(corr_linkage, distance_cutoff, criterion='distance')
    cluster_id_to_feature_ids = defaultdict(list)
    for idx, cluster_id in enumerate(cluster_ids):
        cluster_id_to_feature_ids[cluster_id].append(idx)

    if gini_impurities is None:
        corr_labels = []
        for feature in features_df.columns:
            corr_ = spearmanr(features_df[feature], labels).correlation
            corr_labels.append(corr_)
        corr_labels = np.asarray(corr_labels)
        selected_features = [v[np.argmax(abs(corr_labels[v]))]  for v in cluster_id_to_feature_ids.values()]
    else:
        selected_features = [v[np.argmax(gini_impurities[v])] for v in cluster_id_to_feature_ids.values()]


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    dendro = hierarchy.dendrogram(corr_linkage, 
                            labels=rename_features(features_df.columns), ax=ax1,
                                            leaf_rotation=90)
    ax1.axhline(y=distance_cutoff, color='black', linestyle='dashed')
    dendro_idx = np.arange(0, len(dendro['ivl']))


    im = ax2.imshow(corr[dendro['leaves'], :][:, dendro['leaves']])
    fig.colorbar(im, orientation='horizontal', pad = 0.25)
    ax2.set_xticks(dendro_idx)
    ax2.set_yticks(dendro_idx)
    ax2.set_xticklabels(dendro['ivl'], rotation='vertical')
    ax2.set_yticklabels(dendro['ivl'])
    fig.tight_layout()

    if experiment is None:
        return fig, features_df[features_df.columns[selected_features].values]

    else:
        experiment.log_figure(figure_name="Clustering", figure=fig)
        return features_df[features_df.columns[selected_features].values]

