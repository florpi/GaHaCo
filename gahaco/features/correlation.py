import pandas as pd
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
from collections import defaultdict

def select_uncorrelated_features(features_df):
	'''
	Clusters the Spearman rank-roder correlation of the different features, we keep only a single feature per cluster

	Args:
		features_df: pandas data frame of features
	Returns:
		reduced dataframe with only low correlated features
	'''
	print(features_df.columns)
	corr = spearmanr(features_df).correlation
	corr_linkage = hierarchy.ward(corr)
	cluster_ids = hierarchy.fcluster(corr_linkage, 1., criterion='distance')
	cluster_id_to_feature_ids = defaultdict(list)
	for idx, cluster_id in enumerate(cluster_ids):
		cluster_id_to_feature_ids[cluster_id].append(idx)
	selected_features = [v[0] for v in cluster_id_to_feature_ids.values()]

	selected_features.append(10)

	return features_df[features_df.columns[selected_features].values]
