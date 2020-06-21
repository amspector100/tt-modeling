import pandas as pd
import numpy as np
from scipy import stats

def generate_transition_matrix(bins, ro, mu, sigma):
    mat = np.zeros((len(bins),)*2)
    n = stats.norm(loc=mu, scale=sigma*ro)
    ro_fact = np.sqrt(1-ro**2)
    for i, b1 in enumerate(bins):
        for j, b2 in enumerate(bins):
            mat[i,j] = n.pdf(b2 - ro_fact*b1)
    return mat / mat.sum()

def generate_node_potentials(df, bins):
    pts_t = np.array(df['point']).reshape((len(df),1))*2 - 1
    bins_t = bins.reshape((1,bins.shape[0]))
    return 1/(1 + np.exp(-pts_t*bins_t))