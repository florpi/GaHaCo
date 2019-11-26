import numpy as np
from scipy.special import erf
from scipy.stats import binned_statistic
from scipy.optimize import curve_fit

# Basic HOD model for central galaxies

class HOD():

    def __init__(self, log_m200c, N_gals, satellites=False):

        self.m200c = 10**log_m200c
        n_centrals = N_gals > 0 
        print(n_centrals)

        nbins = 20
        mass_bins = np.logspace(np.log10(np.min(self.m200c)), np.log10(np.max(self.m200c)), 
                nbins + 1)
        self.mass_c = 0.5 * (mass_bins[1:] + mass_bins[:-1])

        self.measured_n_central, _, _ = binned_statistic(self.m200c, n_centrals, 
                                                    statistic = 'mean',
                                                    bins=mass_bins)

        logMmin, sigma_logM = self.fit_hod_centrals(self.mass_c, 
                self.measured_n_central)

        self.hod_parameters = {
                                'logMmin': logMmin,
                                'sigma_logM': sigma_logM,
                              }
        self.mean_n_central = self.mean_occupation_centrals(self.mass_c, **self.hod_parameters)

        if satellites:
            self.measured_n_satellites, _, _ = binned_statistic(self.m200c, N_gals-1, 
                                                    statistic = 'mean',
                                                    bins=mass_bins)


            M0, M1, alpha = self.fit_hod_satellites(self.mass_c, 
                self.measured_n_satellites)

            self.hod_parameters['M0'] = M0
            self.hod_parameters['M1'] = M1
            self.hod_parameters['alpha'] = alpha



    def mean_occupation_centrals(self, halo_mass, logMmin, sigma_logM):

        return 0.5*( 1 + erf((np.log10(halo_mass) - logMmin)/sigma_logM) )

    def mean_occupation_satellites(self, halo_mass, sigma_logM, M0, M1, alpha):
        return self.mean_occupation_centrals(self, halo_mass,  self.logMin, self,sigma_logM) * \
                ((halo_mass - M0)/M1)**alpha

    def n_central(self):
        return self.mean_occupation_centrals(self.halo_mass, self.logMmin, self.sigma_logM)
    def n_satellites(self):
        return self.mean_occupation_satellites(self.halo_mass, self.M0, self.M1, alpha)

    def fit_hod_centrals(self, halo_mass, mean_n_central):

        popt_central, pcov_central = curve_fit(self.mean_occupation_centrals, 
                                                halo_mass, mean_n_central,
                                                p0 = (12., 0.2))
        return popt_central

    def fit_hod_satellites(self, halo_mass, mean_n_central):

        popt_sat, pcov_sat= curve_fit(self.mean_occupation_satellites, 
                                                halo_mass, mean_n_sat,
                                                p0 = (12., 0.2, 1.))

        return popt_sat 



    def populate_centrals(self):
        if(np.max(self.m200c) < 1.e5):
            self.m200c = 10**self.m200c
        np.random.seed(22222)
        Udf = np.random.uniform(0,1,len(self.m200c))
        n_centrals = (self.mean_occupation_centrals(self.m200c,
                        **self.hod_parameters) > Udf).astype(int)
        return n_centrals
