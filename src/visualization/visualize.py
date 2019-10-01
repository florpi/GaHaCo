import numpy as np
import matplotlib.pyplot as plt

from GaHaCo.src.utils.tpcf import compute_tpcf
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels



def plot_confusion_matrix(y_true, y_pred, classes,
						  normalize=False,
						  title=None,
						  cmap=plt.cm.Blues, 
						  experiment = None,
						  log_name = 'Confusion Matrix'):
	"""
	This function prints and plots the confusion matrix.
	Normalization can be applied by setting `normalize=True`.
	"""
	if not title:
		if normalize:
			title = 'Normalized confusion matrix'
		else:
			title = 'Confusion matrix, without normalization'

	# Compute confusion matrix
	cm = confusion_matrix(y_true, y_pred)
	# Only use the labels that appear in the data
	#classes = classes[unique_labels(y_true, y_pred)]
	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		print("Normalized confusion matrix")
	else:
		print('Confusion matrix, without normalization')


	fig, ax = plt.subplots()
	im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
	ax.figure.colorbar(im, ax=ax)
	# We want to show all ticks...
	ax.set(xticks=np.arange(cm.shape[1]),
		   yticks=np.arange(cm.shape[0]),
		   # ... and label them with the respective list entries
		   xticklabels=classes, yticklabels=classes,
		   title=title,
		   ylabel='True label',
		   xlabel='Predicted label')

	# Rotate the tick labels and set their alignment.
	plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
			 rotation_mode="anchor")

	# Loop over data dimensions and create text annotations.
	fmt = '.2f' if normalize else 'd'
	thresh = cm.max() / 2.
	for i in range(cm.shape[0]):
		for j in range(cm.shape[1]):
			ax.text(j, i, format(cm[i, j], fmt),
					ha="center", va="center",
					color="white" if cm[i, j] > thresh else "black")
	fig.tight_layout()
	if experiment is not None:
		experiment.log_figure(figure_name = log_name, figure = fig)
	else:
		return cm, ax


def plot_feature_importance(rf: None,
		feature_names: list,
		experiment = None):
	'''
	Function to generate a bar plot that shows the feature importance after training

	Args:
		rf: Random Forest object.
		feature_names: list containing a readable name for the features.
		experiment: if None, the plot is returned and not logged in, if comet ML experiment object
		given, the plot is logged.

	'''

	importances = rf.feature_importances_
	std = np.std([tree.feature_importances_ for tree in rf.estimators_],
								 axis=0)
	indices = np.argsort(importances)[::-1]
	feature_names = np.array(feature_names)

	fig = plt.figure()

	plt.bar(range(len(feature_names)), importances[indices],
		               color="r", yerr=std[indices], align="center")
	plt.xticks(range(len(feature_names)), feature_names[indices])
	plt.xlim([-1, len(feature_names)])

	if experiment is not None:
		experiment.log_figure(figure_name = 'Feature importance', figure = fig)

	else:
		return fig


def plot_tpcf(pred_positions, label_positions, experiment = None):
	'''
	Plots the ratio of the predicted correlation function to the simulation's one.

	Args:
		pred_positions: positions of the luminous objects predicted by the model.
		label_positions: positions of the luminous objects in the simulation.
		experiment: comet ml experiment to log the figure.


	'''
	
	r_c, pred_tpcf = compute_tpcf(pred_positions)
	r_c, label_tpcf = compute_tpcf(label_positions)

	fig = plt.figure()
	plt.plot(r_c, (pred_tpcf/label_tpcf), label = 'Random Forest')

	plt.axhline(y = 1., color='gray', linestyle='dashed')
	#plt.legend()
	plt.ylim(0.8,1.2)
	plt.ylabel(r'$\hat{\xi}/\xi_{sim}$')
	plt.xlabel(r'$r$ [Mpc/h]')

	
	if experiment is not None:
		experiment.log_figure(figure_name = '2PCF', figure = fig)

	else:
		return fig

		






