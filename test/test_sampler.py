import numpy as np
import scipy as sp
import unittest
from .context import modeling

from modeling import processing, sampler

class TestAR1HMM(unittest.TestCase):

	def simple_test_sampler(self):

		df = 

	def test_sampler(self):

		df = processing.process_data()
		bins = np.round(np.arange(-2.5,2.6,.1), 3)
		X = sampler.sample_conditional(
			df=df,
			bins=bins, 
			mu=0, 
			sigma=0.1, 
			rhos=np.array([0.5, 0.5, 0.5]),
			n_sample=10,
			verbose=True
		)

if __name__ == '__main__':
	unittest.main()