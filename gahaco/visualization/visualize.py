import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from gahaco.utils.tpcf import compute_tpcf, compute_power_spectrum
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import (r2_score, mean_squared_error, confusion_matrix)
from scipy.stats import binned_statistic

def rename_features(feature_names):
    RENAME_DICT = {'M200c':r'$M_{200c}$', 'R200c':r'$R_{200c}$','VelDisp': r'$\sigma_v$',
                'concentration_prada': r'$c_{prada}$', 'Rhosnfw': r'$\rho_{NFW}$',
                'env_5': r'$\Delta M_5$', 'chisq_nfw': r'$\chi^2_{NFW}$',
                'vpeak': r'$V_{peak}$', 'rho_s': r'$\rho_s$', 'HalfmassRad': r'$r_{1/2}$',
                'N_subhalos': r'Nsubhalos',
                'env_10': r'$\Delta M_{10}$', 'concentration_nfw': r'$c_{NFW}$',
                'vel_ani_param': r'$\beta_v$', 'fsub_unbound':'unbound fraction',
                'Vmax':r'$V_{max}$', 'Rmax':r'$R_{max}$'} 

    feature_names = [RENAME_DICT.get(c, c) for c in feature_names] 

    return feature_names

def plot_confusion_matrix(
    cms,
    classes=['Dark', 'Luminous'],
    normalize=False,
    title=None,
    cmap=plt.cm.Blues,
    experiment=None,
    stellar_mass_thresholds = [9]
):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    cm = np.mean(cms, axis=0)
    std_cm = np.std(cms, axis=0)

    if len(cm.shape) < 3:
        cm = cm[np.newaxis,...]
        std_cm = std_cm[np.newaxis,...]

    for k, cm_mass_threshold in enumerate(cm):
        fig, ax = plt.subplots()
        im = ax.imshow(cm_mass_threshold, interpolation="nearest", cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        ## We want to show all ticks...
        ax.set(
            xticks=np.arange(cm_mass_threshold.shape[1]),
            yticks=np.arange(cm_mass_threshold.shape[0]),
        #    # ... and label them with the respective list entries
            xticklabels=classes,
            yticklabels=classes,
            title=f'{title} - $M_\star > $ {10**stellar_mass_thresholds[k]:.1E}',
            ylabel="True label",
            xlabel="Predicted label",
        )

        ## Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        ## Loop over data dimensions and create text annotations.
        fmt = ".2f" if normalize else "d"
        thresh = cm_mass_threshold.max() / 2.0
        for i in range(cm_mass_threshold.shape[0]):
            for j in range(cm_mass_threshold.shape[1]):
                ax.text(
                    j,
                    i,
                    f'${cm_mass_threshold[i,j]:.2f} \pm {std_cm[k,i,j]:.2E}$',
                    #format(cm_mass_threshold[i, j], fmt),
                    ha="center",
                    va="center",
                    color="white" if cm_mass_threshold[i, j] > thresh else "black",
                )
        fig.tight_layout()
        if experiment is not None:
            experiment.log_figure(figure_name="Confusion Matrix", figure=fig)

def plot_feature_importance(list_importances,
        feature_names,
        title='Feature importance',
        experiment=None):
    feature_names = rename_features(feature_names)

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


def plot_tpcfs(r_c,
        hydro_tpcfs, 
        pred_tpcfs, 
        hod_tpcfs,
        experiment=None,
        stellar_mass_thresholds = [9]):
    """
    Plots the ratio of the predicted correlation function to the simulation's one.

    Args:
        experiment: comet ml experiment to log the figure.
    """

    hydro_mean_folds = np.mean(hydro_tpcfs, axis=0)
    hydro_std_folds = np.std(hydro_tpcfs, axis=0)

    if pred_tpcfs is not None:
        pred_mean_folds = np.mean(pred_tpcfs, axis=0)
        pred_std_folds = np.std(pred_tpcfs, axis=0)

    hod_mean_folds = np.mean(hod_tpcfs, axis=0)
    hod_std_folds = np.std(hod_tpcfs, axis=0)


    if len(hydro_mean_folds.shape) < 2:
        hydro_mean_folds = hydro_mean_folds[np.newaxis,...]
        hydro_std_folds = hydro_std_folds[np.newaxis,...]
        if pred_tpcfs is not None:
            pred_mean_folds = pred_mean_folds[np.newaxis,...]
            pred_std_folds  = pred_std_folds [np.newaxis,...]
        hod_mean_folds = hod_mean_folds[np.newaxis,...]
        hod_std_folds  = hod_std_folds [np.newaxis,...]

    height_ratios = [4 + len(hydro_std_folds)] + len(hydro_std_folds)*[1]
    fig, axes = plt.subplots(nrows = hydro_mean_folds.shape[0] + 1, ncols = 1, sharex = True,
                          gridspec_kw = {'height_ratios':height_ratios}, figsize=(8,12))


    colors = ['black', 'midnightblue', 'indianred']
    for i, hdyro_mass in enumerate(hydro_std_folds):
        axes[0].errorbar(r_c, r_c**2*(hydro_mean_folds[i])+ i*50 , yerr = r_c**2*hydro_std_folds[i],
                label = f'TNG', color = colors[i], linestyle='', marker='o', markersize=2)
        if pred_tpcfs is not None:
            axes[0].errorbar(r_c, r_c**2*(pred_mean_folds[i]) + i*50, yerr = r_c**2*pred_std_folds[i],
                    label = f'LGBM', color = colors[i])
        axes[0].errorbar(r_c, r_c**2*(hod_mean_folds[i])+i*50, label = f'HOD', color = colors[i],
                yerr = r_c**2*hod_std_folds[i], linestyle='dashed')
        axes[0].set_ylabel(r'$r^2{\xi}(r)$')

        axes[0].annotate(f'$M_\star > $ {10**stellar_mass_thresholds[i]:.1E}', (15, 20 + i*50), color=colors[i])

        axes[i+1].axhline(y = 0., color='gray', linestyle='dashed')
        axes[i+1].axhline(y = -1., color='gray', linestyle='dashed')
        axes[i+1].axhline(y = 1., color='gray', linestyle='dashed')
 
        if pred_tpcfs is not None:
            axes[i+1].plot(r_c, (pred_mean_folds[i]- hydro_mean_folds[i])/hydro_std_folds[i],
                    color = colors[i])
            axes[i+1].fill_between(r_c, 
                        (pred_mean_folds[i]-pred_std_folds[i]- hydro_mean_folds[i])/hydro_std_folds[i],
                        (pred_mean_folds[i]+pred_std_folds[i]- hydro_mean_folds[i])/hydro_std_folds[i],
                        color = colors[i], alpha = 0.3)
        axes[i+1].plot(r_c, (hod_mean_folds[i]- hydro_mean_folds[i])/hydro_std_folds[i], color = colors[i],
                linestyle='dashed')
        axes[i+1].fill_between(r_c, (hod_mean_folds[i]-hod_std_folds[i]- hydro_mean_folds[i])/hydro_std_folds[i],
                        (hod_mean_folds[i]+hod_std_folds[i]- hydro_mean_folds[i])/hydro_std_folds[i],
                        color = colors[i], alpha = 0.3)
     
        axes[i+1].set_ylim(-5,5)
        axes[i+1]. set_ylabel(r"$\hat{\xi}/\xi_{sim}$")

    axes[-1].set_xlabel(r'$r$ [Mpc/h]')
    #axes[0].legend()

    if experiment is not None:
        experiment.log_figure(figure_name="2PCF", figure=fig)

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

def regression_metrics(y_label, y_pred):

    return (mean_squared_error(y_label, y_pred), r2_score(y_label,y_pred))

def regression(y_label, y_pred, r2score, mse, fold=0, experiment=None):

    h = sns.jointplot(x=y_label, y=y_pred, kind='hex')
    h.set_axis_labels(
        '$log(M_{*, target}) \,\, [M_\odot]$',
        '$log(M_{*, pred}) \,\, [M_\odot]$',
        fontsize=16
    )
    h.ax_joint.text(8, 12, f"$R^2 = {r2score:.2f}$ ", fontsize = 13)
    h.ax_joint.text(8, 11.5, f"MSE$ = {mse:.3f}$ ", fontsize = 13)


    x0, x1 = h.ax_joint.get_xlim()
    y0, y1 = h.ax_joint.get_ylim()
    lims = [max(x0, y0), min(x1, y1)]
    h.ax_joint.plot(lims, lims, ':k') 
    #h = h.annotate(regression_metrics,  template="{stat[0]}: {val[0]:.2f}' {stat[1]}: {val[1]:.2f}",
    #        stat = ("MSE", '$R^2$'), loc = "upper left", fontsize=12)
    h.fig.tight_layout()
    if experiment is not None:
        experiment.log_figure(figure_name="Regression", figure=h.fig)
    else:
        return h 


def histogram(y_test, y_pred, experiment=None):

    fig = plt.figure()
    counts = plt.hist(y_test, log=True, alpha=0.3, label='label')
    plt.hist(y_pred, log=True, alpha=0.3, bins=counts[1], label='prediction')
    plt.legend()

    if experiment is not None:
        experiment.log_figure(figure_name="Histogram", figure=fig)
    else:
        return fig


def mean_halo_occupation(halo_occ, experiment = None):

    # IllustirsTNG Measurements 
    mean_measured_n_central = np.mean([h_occ.measured_n_central for h_occ in halo_occ], axis=0)
    mean_measured_n_satellites = np.mean([h_occ.measured_n_satellites for h_occ in halo_occ], axis=0)
    
    # HOD Predictions
    mean_n_central = np.mean([h_occ.mean_n_central for h_occ in halo_occ], axis=0)
    mean_n_satellites = np.mean([h_occ.meaan_n_satellites for h_occ in halo_occ], axis=0)
    
    # RNF Predictions
    rnf_pred_dir = "/cosma/home/dp004/dc-cues1/GaHaCo/models/lightgbm_reg/"
    first = True
    for rnf_file in glob.glob(rnf_pred_dir + "test_results_fold?_uncorrelated.h5"):
        if first:
            df_pred = pd.read_hdf(
                rnf_pred_dir + "test_results_fold0_uncorrelated.h5", key='df'
            )
            first = False
        else:
            df_fold = pd.read_hdf(
                rnf_pred_dir + "test_results_fold0_uncorrelated.h5", key='df'
            )
            df_pred = df_pred.append(df_fold, ignore_index=True)

    data_path = "/cosma7/data/dp004/dc-cues1/tng_dataframes/"
    df_tot = pd.read_hdf(data_path + "merged_dataframe_test.h5", key='df')
    df_tot.sample()

    pd.merge(
        df_pred,
        df_tot[["Formation Time", "M_stars_central", "M200_DMO"]],
        on=["Formation Time"],
        how='inner'
    )


    mean_n_central = np.mean([h_occ.mean_n_central for h_occ in halo_occ], axis=0)
    mean_n_satellites = np.mean([h_occ.meaan_n_satellites for h_occ in halo_occ], axis=0)

    #mass_c = np.linspace(np.min(halo_occ[0].mass_c), np.max(halo_occ[0].mass_c), 100)

    fig = plt.figure()
    ax = plt.axes()
    ax.plot(
        halo_occ[0].mass_c,
        mean_measured_n_central,
        marker='o',
        markersize=2,
        linestyle='',
        color='blue',
        alpha=0.4,
        label='Measured'
    )

    ax.plot(
        halo_occ[0].mass_c,
        mean_n_central,
        color='blue',
        label='HOD - Centrals'
    )

    ax.plot(
        halo_occ[0].mass_c,
        measured_n_satellites,
        linestyle='',
        marker='o',
        markersize=3,
        color='indianred',
        alpha=0.4,
        label='Measured'
    )
    ax.plot(
        halo_occ[0].mass_c, 
        mean_n_central*mean_n_satellites,
        color='indianred',
        label='HOD - Satellites'
    )
    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_ylabel('Number of galaxies')
    ax.set_xlabel(r'$M_{200c}$')
    plt.legend()
