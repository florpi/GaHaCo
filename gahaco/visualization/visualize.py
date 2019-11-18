import numpy as np
import matplotlib.pyplot as plt

from gahaco.utils.tpcf import compute_tpcf, compute_power_spectrum
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from scipy.stats import binned_statistic

RENAME_DICT = {'M200c':r'$M_{200c}$', 'R200c':r'$R_{200c}$','VelDisp': r'$\sigma_v$',
			'concentration_prada': r'$c_{prada}$', 'Rhosnfw': r'$\rho_{s, prada}$',
			'env_5': r'$\Delta M_5$', 'chisq_nfw': r'$\chi^2_{NFW}$',
			'vpeak': r'$V_{peak}$', 'rho_s': r'$\rho_{s, NFW}$', 'HalfmassRad': r'$r_{1/2}$',
			'N_subhalos': r'Nsubhalos',
			'env_10': r'$\Delta M_{10}$', 'concentration_nfw': r'$c_{NFW}$',
			'vel_ani_param': r'$\beta_v$', 'fsub_unbound':'unbound fraction',
			'Vmax':r'$V_{max}$', 'Rmax':r'$R_{max}$'} 


def plot_confusion_matrix(
	cms,
	classes=['Dark', 'Luminous'],
	normalize=False,
	title=None,
	cmap=plt.cm.Blues,
	experiment=None,
):
	"""
	This function prints and plots the confusion matrix.
	Normalization can be applied by setting `normalize=True`.
	"""
	fig, ax = plt.subplots()
	cm = np.mean(cms, axis=0)
	std_cm = np.std(cms, axis=0)
	im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
	ax.figure.colorbar(im, ax=ax)
	## We want to show all ticks...
	ax.set(
		xticks=np.arange(cm.shape[1]),
		yticks=np.arange(cm.shape[0]),
	#	 # ... and label them with the respective list entries
		xticklabels=classes,
		yticklabels=classes,
		title=title,
		ylabel="True label",
		xlabel="Predicted label",
	)

	## Rotate the tick labels and set their alignment.
	plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

	## Loop over data dimensions and create text annotations.
	fmt = ".2f" if normalize else "d"
	thresh = cm.max() / 2.0
	for i in range(cm.shape[0]):
		for j in range(cm.shape[1]):
			ax.text(
				j,
				i,
				f'${cm[i,j]:.2f} \pm {std_cm[i,j]:.2f}$',
				#format(cm[i, j], fmt),
				ha="center",
				va="center",
				color="white" if cm[i, j] > thresh else "black",
			)
	fig.tight_layout()
	if experiment is not None:
		experiment.log_figure(figure_name="Confusion Matrix", figure=fig)
	else:
		return fig

def plot_feature_importance(list_importances,
		feature_names,
		title='Feature importance',
		experiment=None):
	feature_names = [RENAME_DICT.get(c, c) for c in feature_names] 

	mean_importance = np.mean(list_importances, axis=0)
	std_importance = np.std(list_importances, axis=0)
	indices = np.argsort(mean_importance)[::-1]
	feature_names = np.array(feature_names)

	fig = plt.figure()
	plt.title(title)

	plt.bar(
		range(len(feature_names)),
		mean_importance[indices],
		color="r",
		yerr=std_importance[indices],
		align="center",
	)
	plt.xticks(range(len(feature_names)), feature_names[indices], rotation = 'vertical')
	plt.xlim([-1, len(feature_names)])
	plt.tight_layout()
	if experiment is not None:
		experiment.log_figure(figure_name="Dropcol Feature importance", figure=fig)
	else:
		return fig


def compute_plot_tpcf(pred_positions, 
		label_positions, 
		hod_positions,
		experiment=None):
	'''
	Plots the ratio of the predicted correlation function to the simulation's one.

	Args:
		pred_positions: positions of the luminous objects predicted by the model.
		label_positions: positions of the luminous objects in the simulation.
		experiment: comet ml experiment to log the figure.
	'''
	r_c, pred_tpcf = compute_tpcf(pred_positions)
	r_c, label_tpcf = compute_tpcf(label_positions)
	r_c, hod_tpcf = compute_tpcf(hod_positions)
	
	fig, axes = plt.subplots(nrows = 2, ncols = 1, sharex = True,
						  gridspec_kw = {'height_ratios':[4,1]})


	axes[0].plot(r_c, r_c**2*label_tpcf, label = 'Hydro', color = 'black')
	axes[0].plot(r_c, r_c**2*pred_tpcf, label = 'RF', color = 'midnightblue',
			linestyle='dashed')
	axes[0].plot(r_c, r_c**2*hod_tpcf, label = 'HOD', color = 'indianred',
			linestyle='dashed')
	axes[0].set_ylabel(r'$r^2{\xi}(r)$')
	axes[1].plot(r_c, pred_tpcf/label_tpcf, color = 'midnightblue')
	axes[1].plot(r_c, hod_tpcf/label_tpcf, color = 'indianred')

	axes[1].axhline(y = 1., color='gray', linestyle='dashed')
	axes[1].fill_between(x = r_c, y1 = 0.99, y2 = 1.01, color = 'yellow')
	axes[1].set_ylim(0.9,1.1)

	axes[1].set_xlabel(r'$r$ [Mpc/h]')
	axes[1]. set_ylabel(r"$\hat{\xi}/\xi_{sim}$")
	axes[0].legend()

	if experiment is not None:
		experiment.log_figure(figure_name="2PCF", figure=fig)
	else:
		return fig

def plot_tpcfs(r_c,
		hydro_tpcfs, 
		pred_tpcfs, 
		hod_tpcfs,
		experiment=None):
	"""
	Plots the ratio of the predicted correlation function to the simulation's one.

	Args:
		experiment: comet ml experiment to log the figure.
	"""

	fig, axes = plt.subplots(nrows = 2, ncols = 1, sharex = True,
						  gridspec_kw = {'height_ratios':[4,1]})


	fold_linestyle=['-', 'dotted', 'dashed', '-.']
	for fold in range(len(hydro_tpcfs)):
		axes[0].plot(r_c, r_c**2*hydro_tpcfs[fold], label = f'Hydro-Fold{fold}', color = 'black',
				linestyle=fold_linestyle[fold])
		axes[0].plot(r_c, r_c**2*pred_tpcfs[fold], label = f'RF-Fold{fold}', color = 'midnightblue',
				linestyle=fold_linestyle[fold])
		axes[0].plot(r_c, r_c**2*hod_tpcfs[fold], label = f'HOD-Fold{fold}', color = 'indianred',
				linestyle=fold_linestyle[fold])
		axes[0].set_ylabel(r'$r^2{\xi}(r)$')
		axes[1].plot(r_c, pred_tpcfs[fold]/hydro_tpcfs[fold], color = 'midnightblue',
				linestyle=fold_linestyle[fold])
		axes[1].plot(r_c, hod_tpcfs[fold]/hydro_tpcfs[fold], color = 'indianred',
				linestyle=fold_linestyle[fold])

	axes[1].axhline(y = 1., color='gray', linestyle='dashed')
	axes[1].fill_between(x = r_c, y1 = 0.99, y2 = 1.01, color = 'yellow')
	axes[1].set_ylim(0.9,1.1)

	axes[1].set_xlabel(r'$r$ [Mpc/h]')
	axes[1]. set_ylabel(r"$\hat{\xi}/\xi_{sim}$")
	axes[0].legend()

	if experiment is not None:
		experiment.log_figure(figure_name="2PCF", figure=fig)
	else:
		return fig

def plot_power_spectrum(pred_positions,
		label_positions,
		hod_positions,
		experiment=None):
	pred_pk = compute_power_spectrum(pred_positions)
	label_pk = compute_power_spectrum(label_positions)
	hod_pk = compute_power_spectrum(hod_positions)
	
	fig, axes = plt.subplots(nrows = 2, ncols = 1, sharex = True,
						  gridspec_kw = {'height_ratios':[4,1]})


	axes[0].loglog(label_pk['k'], label_pk['power'].real - label_pk.attrs['shotnoise'],
						   label = 'Hydro', color = 'black')
	axes[0].loglog(pred_pk['k'], pred_pk['power'].real - pred_pk.attrs['shotnoise'],
						   label = 'RNF', color = 'midnightblue', 
						   linestyle='dashed')
	axes[0].loglog(hod_pk['k'], hod_pk['power'].real - hod_pk.attrs['shotnoise'],
						   label = 'HOD', color = 'indianred',
						   linestyle='dashed')

	axes[1].semilogx(hod_pk['k'], 
				(hod_pk['power'].real - hod_pk.attrs['shotnoise'])\
							/( label_pk['power'].real - label_pk.attrs['shotnoise']), color = 'indianred')
	axes[1].semilogx(pred_pk['k'], 
				(pred_pk['power'].real - pred_pk.attrs['shotnoise'])\
							/( label_pk['power'].real - label_pk.attrs['shotnoise']), color = 'midnightblue')


	axes[1].axhline(y = 1., color='gray', linestyle='dashed')
	axes[1].axhline(y = 1.1, color='gray', linestyle='dashed')
	axes[1].axhline(y = 0.9, color='gray', linestyle='dashed')
	axes[1].fill_between(x = label_pk['k'], y1 = 0.99, y2 = 1.01, color = 'yellow')
	axes[1].set_ylim(0.8,1.2)
	axes[0].set_ylabel(r"$P(k)$ [$h^{-3}\mathrm{Mpc}^3$]")
	axes[1].set_ylabel(r'$\hat{P}}/P_{Hydro}$')

	axes[1].set_xlabel(r"$k$ [$h \ \mathrm{Mpc}^{-1}$]")
	axes[0].legend()
	plt.xlim(0.1, 5)

	if experiment is not None:
		experiment.log_figure(figure_name="2PCF", figure=fig)
	else:
		return fig


def plot_roc_curve(rf,
		X_test,
		y_test,
		experiment = None):
	fig = plt.gca()
	rfc_disp = plot_roc_curve(rfc, X_test, y_test, ax=fig, alpha=0.8)
	
	if experiment is not None:
		experiment.log_figure(figure_name="2PCF", figure=fig)

	else:
		return fig

def halo_occupation(halo_mass, n_gals, mass_bins):
	mean_gals_per_mass, _, _ = binned_statistic(halo_mass, n_gals, 
												statistic = 'mean',
												bins=mass_bins)


	std_gals_per_mass, _, _ = binned_statistic(halo_mass, n_gals, 
												statistic = 'std',
												bins=mass_bins)

	return mean_gals_per_mass, std_gals_per_mass



def plot_halo_occupation(halo_mass, n_gals_label,
		n_gals_pred, n_gals_hod, experiment=None):

	nbins = 20
	mass_bins = np.logspace(np.log10(np.min(halo_mass)), np.log10(np.max(halo_mass)), nbins + 1)
	mass_c = 0.5 * (mass_bins[1:] + mass_bins[:-1])

	mean_gals_label, std_gals_label = halo_occupation(halo_mass, n_gals_label, mass_bins)
	mean_gals_pred, std_gals_pred = halo_occupation(halo_mass, n_gals_pred, mass_bins)
	mean_gals_hod, std_gals_hod = halo_occupation(halo_mass, n_gals_hod, mass_bins)

	fig = plt.figure()
	ax = plt.axes()
	ax.plot(mass_c, mean_gals_label,
				   linestyle = '', marker = 'o', markersize = 3,
						  color = 'gray', alpha = 0.4, label = 'Label')

	ax.fill_between(mass_c, mean_gals_label-std_gals_label,
						  mean_gals_label+std_gals_label, alpha = 0.1,
										  color='gray')
	ax.plot(mass_c, mean_gals_pred,
				   linestyle = '', marker = 'o', markersize = 3,
						  color = 'indianred', alpha = 0.4, label = 'Prediction')

	ax.fill_between(mass_c, mean_gals_pred-std_gals_pred,
						  mean_gals_pred+std_gals_pred, alpha = 0.1,
										  color='indianred')

	ax.plot(mass_c, mean_gals_hod,
				   linestyle = '', marker = 'o', markersize = 3,
						  color = 'royalblue', alpha = 0.4, label = 'HOD')

	ax.fill_between(mass_c, mean_gals_hod-std_gals_hod,
						  mean_gals_hod+std_gals_hod, alpha = 0.1,
										  color='royalblue')


	ax.set_xscale("log")
	ax.set_ylabel('Number of galaxies')
	ax.set_xlabel(r'$M_{200c}$')
	plt.legend()

	if experiment is not None:
		experiment.log_figure(figure_name="hod", figure=fig)
	else:
		return fig


def regression(y_label, y_pred, r2score, fold=0, experiment=None):

	fig = plt.figure()
	plt.plot(y_label, y_label, linestyle='dashed')
	plt.plot(y_label, y_pred, linestyle='', marker='o', markersize=2)
	plt.text(0.5,0.5,f'$R^2$ = {r2score:.4f}')
	plt.xlabel('Target')
	plt.ylabel('Prediction')

	if experiment is not None:
		experiment.log_figure(figure_name="Regression", figure=fig)
	else:
		return fig


