import numpy as np
import matplotlib.pyplot as plt

<<<<<<< HEAD
def halo_mass_function(mass, nbins=20):
    bins = np.logspace(11, np.log10(np.max(mass)), nbins+1)
    mass_count, edges = np.histogram(mass, bins=bins)
    mass_bin = (edges[1:]+edges[:-1])/2.
    return mass_count, mass_bin

    
def plot_halo_mass_function(mass, nbins=20, color='black', label='default'):
    
    if isinstance(mass, numpy.ndarray):
        mass_count, mass_bin = halo_mass_function(mass)
        plt.loglog(
            mass_bin,
            mass_count,
            marker='o',
            markersize=3.,
            color=color,
            label=label
        )

    elif isinstance(mass, list):
        # Run through mass-functions
        for mm in mass:
            mass_func, edges = np.histogram(mm, bins=bins)
            plt.loglog(
                mass_bin,
                mass_count,
                marker='o',
                markersize=3.,
                color=color,
                label=label
            )
        
    plt.ylabel('Number of halos')
    plt.xlabel(r'$M_{200c}$')
=======
def plot_halo_mass_function(mass, color = 'black', label = 'default'):
    nbins= 20
    bins = np.logspace(11,
                       np.log10(np.max(mass)), nbins+1)

    bin_centers = (bins[1:]+bins[:-1])/2.

    mass_func, edges = np.histogram(mass, bins=bins)
    plt.loglog((edges[1:]+edges[:-1])/2.,
               mass_func, marker='o',markersize=3.,
              color = color, label = label)

    plt.ylabel('Number of halos')
    plt.xlabel(r'$M_{200c}$')

>>>>>>> florpi
