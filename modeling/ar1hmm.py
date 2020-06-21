import torch
import numpy as np
from scipy import stats

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
	:param X: n length numpy array of hidden states.
	:param y: n length binary numpy array of emissions
	:param transition_types: n - 1 length array of transition 
	types. Its unique values should be k successive integers,
	zero-indexed.
	:param rhos: A k-length array of correlation constants
	rho. rho[i] corresponds to transition_type == i.
	:param mu: The mean parameter
	:param sigma: The variance / noise parameter
	"""

	# Make sure dimensions are correct
	n = X.shape[0]
	if n != y.shape[0]:
		raise ValueError(
			f"Hidden states shape {X.shape} should equal emisson shape {y.shape}"
		)
	n_trans = transition_types.shape[0]
	if n_trans != n - 1:
		raise ValueError(
			f"Transition types shape {n_trans} should be one less than hidden shape {X.shape}"
		)


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
	differences = X[1:] - ordered_conjugates * X[:-1]
	l_transition = np.log(norm_pdf((differences / ordered_rhos))).sum()

	return l_emission + l_init + l_transition

class EM_Optimizer():

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
		pass