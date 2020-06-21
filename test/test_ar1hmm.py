import numpy as np
import scipy as sp
import unittest
from .context import modeling

from modeling import ar1hmm

class TestAR1HMM(unittest.TestCase):

	def test_log_likelihood(self):

		# Dimensionality
		np.random.seed(100)
		p = 100

		# Fake data
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

if __name__ == '__main__':
	unittest.main()