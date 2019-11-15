import numpy as np
import matplotlib.pyplot as plt

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

