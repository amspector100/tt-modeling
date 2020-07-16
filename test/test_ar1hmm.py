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

		# Fake data
		np.random.seed(123456)
		n = 100
		p = 10
		X = np.random.randn(n, p)
		y = np.random.binomial(1, 0.1, (p,))
		transition_types = np.random.binomial(1, 0.3, (p-1,))

		# Initialize and calc loglike
		torch_ll = ar1hmm.TorchLogLikelihood(y=y, transition_types=transition_types)
		loglike = torch_ll(X)[0].item()

		# Extract parameters 
		mu, sigma, rhos = torch_ll.get_params()
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

	def test_M_step(self):

		# Fake data
		np.random.seed(123456)
		n = 100
		p = 10
		X = np.random.randn(n, p)
		y = np.random.binomial(1, 0.1, (p,))
		transition_types = np.random.binomial(1, 0.3, (p-1,))

		# Class init and initial log-likelihood
		emopt = ar1hmm.EMOptimizer(y=y, transition_types=transition_types)
		init_ll = emopt.model(X).sum().item()

		# M_step
		qloss = emopt.M_step(X, num_iter=50)
		self.assertTrue(
			init_ll < qloss,
			f"M step fails to increase likelihood: initial likelihood is {init_ll}, ending is {qloss}"
		)
	
	def test_EM(self):

		# Fake data
		np.random.seed(123456)
		p = 3000
		X = 0.3*np.random.randn(p) + 0.06
		y = np.random.binomial(1, 1/(1+np.exp(-1*X)), (p,))
		transition_types = np.random.binomial(1, 0.3, (p-1,))

		# Class init and initial log-likelihood
		emopt = ar1hmm.EMOptimizer(
			y=y,
			transition_types=transition_types
		)
		emopt.train(num_iter=2, n_sample=100, verbose=1)
		
		# Makes sure loss is decreasing
		qlosses = emopt.qlosses
		self.assertTrue(
			qlosses[0] < qlosses[-1],
			f"Initial qloss {qlosses[0]} is >= final qloss {qlosses[-1]}"
		)

if __name__ == '__main__':
	unittest.main()