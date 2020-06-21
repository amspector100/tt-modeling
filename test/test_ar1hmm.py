import numpy as np
import scipy as sp
import unittest
from .context import modeling

from modeling import ar1hmm

class TestAR1HMM(unittest.TestCase):

	def test_log_likelihood(self):

		# Fake data
		np.random.seed(123456)
		p = 100
		X = np.random.randn(p)
		y = np.random.binomial(1, 0.1, (p,))
		mu = 0
		sigma = 1
		rhos = np.array([0.2, 0.5])
		transition_types = np.random.binomial(1, 0.5, (p-1,))

		# Calculate log-likelihood
		loglike1 = ar1hmm.AR1_log_likelihood(
			X=X,
			y=y,
			transition_types=transition_types,
			rhos=rhos,
			mu=mu,
			sigma=sigma
		)
		loglike2 = ar1hmm.AR1_log_likelihood(
			X=X - 0.5,
			y=y,
			transition_types=transition_types,
			rhos=rhos,
			mu=mu - 0.5,
			sigma=sigma
		)

		self.assertTrue(
			loglike1 < loglike2,
			"Unexpected behavior from ar1hmm loglikelihood function"
		)

	def test_ar1hmm_class(self):

		# Dimensionality
		np.random.seed(123456)
		n = 100
		p = 10

		# Fake data
		X = np.random.randn(n, p)
		y = np.random.binomial(1, 0.1, (p,))
		transition_types = np.random.binomial(1, 0.3, (p-1,))

		# Initialize and calc loglike
		emopt = ar1hmm.EMOptimizer(y=y, transition_types=transition_types)
		loglike = emopt(X)[0].item()

		# Extract parameters 
		mu, sigma, rhos = emopt.get_params()
		mu = mu.item()
		sigma = sigma.item()
		rhos = rhos.detach().numpy()

		# Check that the loglikelihoods agree
		loglike_numpy = ar1hmm.AR1_log_likelihood(
			X=X[0],
			y=y,
			transition_types=transition_types,
			rhos=rhos,
			mu=mu,
			sigma=sigma
		)
		self.assertTrue(
			abs(loglike-loglike_numpy) < 0.1,
			f"Numpy ({loglike_numpy}) and torch (({loglike}) log likelihoods disagree"
		)

if __name__ == '__main__':
	unittest.main()