import pandas as pd
import numpy as np

def generate_transition_matrix(bins, ro, mu, sigma):
    """
    the parameter sigma is sigma.
    """
    mat = np.zeros((len(bins),)*2)
    conjugate = np.sqrt(1-ro**2)
    denom = 2*sigma**2
    for i, b1 in enumerate(bins):
        for j, b2 in enumerate(bins):
            diff = (b2 - ro*b1) / conjugate
            mat[i,j] = np.exp(-((diff - mu)**2/denom))
    return mat / mat.sum()

def generate_node_potentials(y, bins, mu, sigma):
    """
    :param y: boolean emissions of points. p length numpy array
    """

    # Generate standard node potentials
    pts_t = 2*y - 1
    node_potentials = 1/(1 + np.exp(-1*np.outer(pts_t, bins)))

    # For the first node, since it is normal, we multiply by
    # the normal PDF as well.
    node_potentials[0] = node_potentials[0] * np.exp(-((bins - mu)**2/2*sigma**2))
    return node_potentials