import numpy as np
from scipy.special import erf
from scipy.stats import binned_statistic
from scipy.optimize import curve_fit

# Basic HOD model for central galaxies

class HOD():

	def __init__(self, df, labels):

		self.df = df
		m200c = 10**df['M200c']
		n_centrals = labels 

		nbins = 20
		mass_bins = np.logspace(np.log10(np.min(m200c)), np.log10(np.max(m200c)), 
				nbins + 1)
		self.mass_c = 0.5 * (mass_bins[1:] + mass_bins[:-1])

		self.measured_n_central, _, _ = binned_statistic(m200c, n_centrals, 
													statistic = 'mean',
													bins=mass_bins)

		logMmin, sigma_logM = self.fit_hod_parameters(self.mass_c, 
				self.measured_n_central)

		self.hod_parameters = {
								'logMmin': logMmin,
								'sigma_logM': sigma_logM,
							  }
		self.mean_n_central = self.mean_occupation(self.mass_c, **self.hod_parameters)



	def mean_occupation(self, halo_mass, logMmin, sigma_logM):

		return 0.5*( 1 + erf((np.log10(halo_mass) - logMmin)/sigma_logM) )

	def n_central(self):
		return self.mean_occupation(self.halo_mass, self.logMmin, self.sigma_logM)

	def fit_hod_parameters(self, halo_mass, mean_n_central):

		popt_central, pcov_central = curve_fit(self.mean_occupation, 
												halo_mass, mean_n_central,
												p0 = (12., 0.2))

		return popt_central


	def populate(self):
		self.df.M200c = 10**self.df.M200c
		np.random.seed(22222)
		Udf = np.random.uniform(0,1,len(self.df))
		n_centrals = (self.mean_occupation(self.df.M200c,
			            **self.hod_parameters) > Udf).astype(int)
		return n_centrals
