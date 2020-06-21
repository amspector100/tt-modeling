import numpy as np
import pandas as pd
import scipy as sp
import unittest
from .context import modeling

from modeling import processing, sampler

class TestAR1HMM(unittest.TestCase):

	def test_sampler_marginal_consistency(self):

		# Sample fake hidden state, rho = 0, mu = 0, sigma = 1
		np.random.seed(111)
		n = 50000
		p = 3
		mu = 0
		sigma = 1
		rhos = np.array([0.2, 0.3])
		X_true = np.random.randn(p)
		y_true = np.random.binomial(1, 1 / (1 + np.exp(-X_true)))
		transition_types = np.random.binomial(1, 0.5, p-1).astype(int)
		bins = np.round(np.arange(-2.5,2.6,.1), 3)

		# Pass to sampler
		X = sampler.sample_conditional(
			y=y_true,
			transition_types=transition_types,
			bins=bins, 
			mu=mu, 
			sigma=sigma, 
			rhos=rhos,
			n_sample=n,
			verbose=True
		)
		marginals = sampler.compute_marginals(
			y=y_true, 
			transition_types=transition_types,
			bins=bins,
			mu=mu,
			sigma=sigma,
			rhos=rhos,
		)

		# Check that theoretical means = sampled_means
		theoretical_means = np.dot(marginals, bins)
		sampled_means = X.mean(axis=0)
		np.testing.assert_almost_equal(
			theoretical_means, sampled_means, decimal=2, 
			err_msg=f'Theoretical marginal means {theoretical_means} do not match sampled means {sampled_means}'
		)

	def test_sampler_real_data(self):

		# Real data
		df = processing.process_data()
		y = df['point'].values
		transition_types = df['point_type'][1:]
		bins = np.round(np.arange(-2.5,2.6,.1), 3)
		X = sampler.sample_conditional(
			y=y,
			transition_types=transition_types,
			bins=bins, 
			mu=0, 
			sigma=0.1, 
			rhos=np.array([0.5, 0.5, 0.5]),
			n_sample=10,
			verbose=True
		)

if __name__ == '__main__':
	unittest.main()