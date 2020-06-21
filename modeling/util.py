import pandas as pd
import numpy as np

def generate_transition_matrix(bins, ro, mu, sigma):
    mat = np.zeros((len(bins),)*2)
    ro_fact = np.sqrt(1-ro**2)
    new_sig_sq = (sigma*ro)**2
    for i, b1 in enumerate(bins):
        for j, b2 in enumerate(bins):
            diff = b2 - ro_fact*b1 - mu
            mat[i,j] = np.exp(-(diff**2/(2*new_sig_sq)))
    return mat / mat.sum()

def generate_node_potentials(df, bins):
    pts_t = np.array(df['point']).reshape((len(df),1))*2 - 1
    bins_t = bins.reshape((1,bins.shape[0]))
    return 1/(1 + np.exp(-pts_t*bins_t))