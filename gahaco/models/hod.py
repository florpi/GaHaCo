import numpy as np
from scipy.special import erf
from scipy.stats import binned_statistic
from scipy.optimize import curve_fit

# Basic HOD model for central galaxies

class HOD():

    def __init__(self, m200c, N_gals, satellites=False):

        self.m200c = m200c
        if satellites:
            n_centrals = N_gals > 0 
        else:
            n_centrals = N_gals
        nbins = 20 
        mass_bins = np.logspace(np.log10(np.min(self.m200c)), np.log10(np.max(self.m200c)), 
                nbins + 1)
        self.mass_c = 0.5 * (mass_bins[1:] + mass_bins[:-1])

        self.measured_n_central, _, _ = binned_statistic(self.m200c, n_centrals, 
                                                    statistic = 'mean',
                                                    bins=mass_bins)

        self.logMmin, self.sigma_logM = self.fit_hod_centrals(self.mass_c, 
                self.measured_n_central)

        self.hod_parameters_centrals = {
                                'logMmin': self.logMmin,
                                'sigma_logM': self.sigma_logM,
                              }
        self.mean_n_central = self.mean_occupation_centrals(self.mass_c, **self.hod_parameters_centrals)

        if satellites:
            self.measured_n_satellites, _, _ = binned_statistic(self.m200c, N_gals-1, 
                                                    statistic = 'mean',
                                                    bins=mass_bins)


            logMcut, logM1, alpha = self.fit_hod_satellites(self.mass_c, 
                self.measured_n_satellites)

            self.hod_parameters_sats = {
                                'logMcut': logMcut,
                                'logM1': logM1,
                                'alpha': alpha
                              }
            self.mean_n_satellites = self.mean_occupation_satellites(self.mass_c, **self.hod_parameters_sats)

    def mean_occupation_centrals(self, halo_mass, logMmin, sigma_logM):

        return 0.5*( 1 + erf((np.log10(halo_mass) - logMmin)/sigma_logM) )

    def mean_occupation_satellites(self, halo_mass, logMcut, logM1, alpha):
        Mcut = 10.**logMcut
        M1 = 10.**logM1
        satellites_occ = self.mean_occupation_centrals(halo_mass, **self.hod_parameters_centrals)*((halo_mass - Mcut)/M1)**alpha
        return np.nan_to_num(satellites_occ)

    def n_central(self):
        return self.mean_occupation_centrals(self.halo_mass, self.logMmin, self.sigma_logM)

    def n_satellites(self):
        return self.mean_occupation_satellites(self.halo_mass, self.logMcut, self.logM1, self.alpha)

    def fit_hod_centrals(self, halo_mass, mean_n_central):
        mean_n_central[np.isnan(mean_n_central)] = 1
        popt_central, pcov_central = curve_fit(self.mean_occupation_centrals, 
                                                halo_mass, mean_n_central,
                                                p0 = (11.6, 0.17))
        return popt_central

    def fit_hod_satellites(self, halo_mass, mean_n_sat):
        bounds=([10, 10, 0.5], [13,13, 2.5])
        #threshold = (halo_mass > 1.e12) & (halo_mass < 1.e14)
        threshold = (halo_mass < 1.e14)

        popt_sat, pcov_sat= curve_fit(self.mean_occupation_satellites, 
                                                halo_mass[threshold], mean_n_sat[threshold],
                                                bounds=bounds,
                                                p0 = (11.,12., 0.8))
        return popt_sat 

    def populate_centrals(self, m200c):
        if(np.max(m200c) < 1.e5):
            m200c = 10**m200c
        np.random.seed(22222)
        Udf = np.random.uniform(0,1,len(m200c))
        n_centrals = (self.mean_occupation_centrals(m200c,
                        **self.hod_parameters_centrals) > Udf).astype(int)
        return n_centrals

    def populate_satellites(self, n_centrals):

        n_satellites = np.random.poisson(self.mean_occupation_satellites(self.m200c, **self.hod_parameters_sats),
                                         len(self.m200c))

        exception = (n_centrals == 0) & (n_satellites!=0)
        n_satellites[exception] -= 1
        n_centrals[exception] += 1

        return n_centrals, n_satellites

    def satellites_positions(self, halo_pos, n_satellites, concentration, r200c):

        R_s = r200c/concentration
        fc = np.log(1. + concentration) - concentration / (1. + concentration)
        Rho_s = self.m200c / (4*np.pi*R_s**3 * fc)
        satellite_positions = np.zeros((np.sum(n_satellites), 3))
        satellite_counter = 0
        for halo in range(len(halo_pos)):
            theta  = np.arccos(2.0 * np.random.ranf(n_satellites[halo]) - 1.0)
            phi = 2.0 * np.pi * np.random.ranf(n_satellites[halo])
            M_rand   = self.m200c[halo]* np.random.ranf(n_satellites[halo])
            r = 10.0**(np.arange(-5.0, 0.05, 0.05)) * r200c[halo]
            Mnfw = 4. * np.pi * Rho_s[halo] * R_s[halo]**3 * (np.log(1.+r/R_s[halo]) - (r/R_s[halo])/(1.+r/R_s[halo]))
            rp = np.interp(M_rand, Mnfw, r)
            xp = rp * np.sin(theta) * np.cos(phi)
            yp = rp * np.sin(theta) * np.sin(phi)
            zp = rp * np.cos(theta)

            satellite_positions[satellite_counter:satellite_counter+n_satellites[halo],:] = halo_pos[halo,:] 
            satellite_positions[satellite_counter:satellite_counter+n_satellites[halo],0] +=  xp/1000.
            satellite_positions[satellite_counter:satellite_counter+n_satellites[halo],1] += yp/1000.
            satellite_positions[satellite_counter:satellite_counter+n_satellites[halo],2] += zp/1000.

            satellite_counter += n_satellites[halo]


        return satellite_positions 
