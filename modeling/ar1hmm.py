import time
import numpy as np
from scipy import stats

import torch
import torch.nn as nn
import torch.distributions as ds

from . import sampler 

def Check_Shapes(y, transition_type):
	p = y.shape[0]
	p_trans = transition_type.shape[0]
	if p_trans != p - 1:
		raise ValueError(
			f"Transition types dim 0 {p_trans} should be one less than emission dim 0 {p}"
		)


def AR1_log_likelihood(
	X,
	y,
	transition_types,
	rhos,
	mu,
	sigma
):
	"""
	Calculates the likelihood. 
	:param X: p length numpy array of hidden states.
	:param y: p length binary numpy array of emissions
	:param transition_types: p - 1 length array of transition 
	types. Its unique values should be k successive integers,
	zero-indexed.
	:param rhos: A k-length array of correlation constants
	rho. rho[i] corresponds to transition_type == i.
	:param mu: The mean parameter
	:param sigma: The variance / noise parameter
	"""

	# Make sure dimensions are correct
	p = X.shape[0]
	if p != y.shape[0]:
		raise ValueError(
			f"Hidden states shape {X.shape} should equal emisson shape {y.shape}"
		)
	Check_Shapes(y, transition_types)

	# Emission likelihoods
	sigmoids = np.exp(X) / (1 + np.exp(X))
	l_emission = np.log(sigmoids[y == 1]).sum()
	l_emission += np.log((1-sigmoids)[y == 0]).sum()

	# Initial likelihood 
	norm_pdf = stats.norm(loc=mu, scale=sigma).pdf
	l_init = np.log(norm_pdf(X[0]))

	# Transition likelihoods based on AR1 likelihood
	ordered_rhos = rhos[transition_types]
	ordered_conjugates = np.sqrt(1 - np.power(ordered_rhos, 2))
	differences = X[1:] - ordered_rhos * X[:-1]
	l_transition = np.log(norm_pdf((differences / ordered_conjugates))).sum()

	return l_emission + l_init + l_transition

class TorchLogLikelihood(torch.nn.Module):

	def __init__(
		self,
		y,
		transition_types,
	):
		"""
		:param y: n length binary numpy array of emissions
		:param transition_types: n - 1 length array of transition 
		types. Its unique values should be k successive integers,
		zero-indexed.
		"""
		super().__init__()

	
		# Save observed data
		self.p = y.shape[0]
		self.y = y
		self.transition_types = transition_types


		# Initialize parameters
		self.mu = torch.nn.Parameter(torch.tensor(0).float())
		self.sigma = torch.nn.Parameter(torch.tensor(0.2).float())
		num_rhos = np.unique(self.transition_types).shape[0]

		# Because we constrain rhos in [0,1], 
		# we store log(rho) as a the variable we optimize
		init_rho_logit = torch.zeros(num_rhos) - 1
		self.rho_logit = torch.nn.Parameter(init_rho_logit)

	def get_params(self):
		""" Returns parameters, convenience function """
		rhos = torch.sigmoid(self.rho_logit)
		return self.mu, self.sigma, rhos

	def forward(self, X):

		"""
		Calculates the likelihood. 
		:param X: n x p length numpy array of hidden states.
		"""

		# Torch-ify
		X = torch.tensor(X).detach().float()

		# Emission likelihoods
		sigmoids = torch.sigmoid(X)
		mask = torch.tensor(self.y == 1).byte()
		l_emission = torch.log(sigmoids[:, mask]).sum(dim=1)
		l_emission += torch.log((1-sigmoids[:, ~mask])).sum(dim=1)

		# Initial likelihood 
		norm_rv = ds.normal.Normal(loc=self.mu, scale=self.sigma)
		l_init = norm_rv.log_prob(X[:, 0])

		# Transition likelihoods based on AR1 likelihood
		rhos = torch.sigmoid(self.rho_logit)
		ordered_rhos = rhos[torch.tensor(self.transition_types).long()]
		ordered_conjugates = torch.sqrt(1 - ordered_rhos**2)
		differences = X[:, 1:] - ordered_rhos * X[:, :-1]
		l_transition = norm_rv.log_prob(differences / ordered_conjugates).sum(dim=1)

		return l_emission + l_init + l_transition

class EMOptimizer():

	def __init__(
		self,
		df=None,
		y=None,
		transition_types=None,
		binsize=0.1,
		max_bin=2.5,
		**kwargs
	):
		"""
		:param y: n length binary numpy array of emissions
		:param transition_types: n - 1 length array of transition 
		types. Its unique values should be k successive integers,
		zero-indexed.
		:param binsize: Size of bins
		:param max_bin: Largest bin value. Defaults to 2.5.
		:param **kwargs: kwargs for adam optimizer.
		"""

		# Extract np-ified data
		if df is not None:
			y = df['point'].values
			transition_types = df['point_type'].values
			transition_types = transition_types[1:].astype(int)

		# Check transition type shape
		Check_Shapes(y, transition_types)

		# Initialize log-likelihood module and save data
		self.p = y.shape[0]
		self.y = y
		self.transition_types = transition_types
		self.model = TorchLogLikelihood(
			y=y,
			transition_types=transition_types,
		)

		# Gradient-based optimizer for M step
		if 'lr' not in kwargs:
			kwargs['lr'] = 0.01
		self.opt = torch.optim.Adam(
			self.model.parameters(),
			**kwargs
		)

		# Bins
		self.bins = np.arange(
			-1*max_bin, max_bin + binsize, binsize
		)

		# Save q losses for later caching
		self.qlosses = []

	def get_params(self):
		"""
		Helper function to pull numpy-ified parameters.
		returns: mu, sigma, rhos
		"""
		mu, sigma, rhos = self.model.get_params()
		mu = mu.item()
		sigma = sigma.item()
		rhos = rhos.detach().numpy()
		return mu, sigma, rhos

	def E_step(self, n_sample=1000, verbose=True):
		"""
		Samples from P(X|y, mu, sigma, rhos) 
		returns: n x p array X of bucketized normal
		values.
		"""
		mu, sigma, rhos = self.get_params()
		X = sampler.sample_conditional(
			y=self.y,
			transition_types=self.transition_types,
			bins=self.bins,
			mu=mu,
			sigma=sigma,
			rhos=rhos,
			n_sample=n_sample,
			verbose=verbose,
		)
		return X

	def M_step(self, X, num_iter=50, mstep=0, verbose=True):
		"""
		:param X: n x p array of hidden states
		(n = number of samples, p = length of chain)
		returns: mu, sigma, rhos
		"""
		for j in range(num_iter):

			# Step 1: Calculate loss
			qloss = -1*self.model(X).sum()

			# Step 2: Step along gradient
			self.opt.zero_grad()
			qloss.backward()
			self.opt.step()

			# Step 3: Log output
			if j % 5 == 0 and verbose:
				print(f"At M step {mstep}, iter {j}, qloss is {-1*qloss.item()}")

		return -1*qloss

	def train(
		self,
		num_iter=10,
		num_M_iter=50,
		n_sample=1000,
		verbose=1
	):
		"""
		:param verbose: If 0, logs nothing. If 1, logs completions
		of EM steps. If > 1, then the M and E steps will be verbose as well.
		returns: mu, sigma, rhos, marginals 
		"""

		# Iterate
		print("Running the EM algorithm... \n")
		time0 = time.time()
		for i in range(num_iter):

			# E step
			X = self.E_step(
				n_sample=n_sample,
				verbose=verbose > 1
			)

			# M step
			qloss = self.M_step(
				X=X,
				num_iter=num_M_iter,
				mstep=i,
				verbose=verbose > 1
			) 

			# Cache for testing / analysis
			self.qlosses.append(qloss)

			if verbose > 0:
				print(f"EM iteration {i} has qloss {qloss} at time {time.time() - time0}")

		# Extract parameters
		print("Finished with EM algorith... extracting marginals!")
		mu, sigma, rhos = self.get_params()
		self.marginals = sampler.compute_marginals(
			y=self.y, 
			transition_types=self.transition_types,
			bins=self.bins,
			mu=mu,
			sigma=sigma,
			rhos=rhos,
		)
		# Return
		return mu, sigma, rhos, self.marginals









