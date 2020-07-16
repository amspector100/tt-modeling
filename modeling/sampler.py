import numpy as np
from tqdm import *

from .util import *
from .processing import *

from random import choices


def compute_fwd_messages(node_potentials, edge_potentials):
    # Make sure shapes, are, like, reasonable
    assert(node_potentials.shape[0] == edge_potentials.shape[0]+1)
    assert(node_potentials.shape[1] == edge_potentials.shape[1])
    assert(edge_potentials.shape[1] == edge_potentials.shape[2])

    # Figure out possible IDs and states
    ids = np.arange(node_potentials.shape[0])
    states = np.arange(node_potentials.shape[1])

    # Messages has following shape:
    # Dim 0 -- nodes in hmm
    # Dim 1 -- bins
    messages = np.empty((ids.shape[0]-1,)+states.shape, dtype=np.float32)
    for value_end in states:
        messages[0][value_end] = np.einsum(
            'i,i->',
            node_potentials[0],
            edge_potentials[0,:,value_end]
        )
    messages[0] /= messages[0].sum()
    for e in ids[1:-1]:
        for value_end in states:
            messages[e][value_end] = np.einsum(
                'i,i,i->',
                node_potentials[e],
                edge_potentials[e,:,value_end],
                messages[e-1]
            )
        messages[e] /= messages[e].sum()
    return messages

def compute_bkw_messages(node_potentials, edge_potentials):
    # Make sure shapes, are, like, reasonable
    assert(node_potentials.shape[0] == edge_potentials.shape[0]+1)
    assert(node_potentials.shape[1] == edge_potentials.shape[1])
    assert(edge_potentials.shape[1] == edge_potentials.shape[2])

    # Figure out possible IDs and states
    ids = np.arange(node_potentials.shape[0])
    states = np.arange(node_potentials.shape[1])

    # Messages has following shape:
    # Dim 0 -- nodes in hmm
    # Dim 1 -- bins
    messages = np.empty((ids.shape[0]-1,)+states.shape, dtype=np.float32)
    for value_end in states:
        messages[0][value_end] = np.einsum(
            'i,i->', node_potentials[-1],
            edge_potentials[-2,value_end,:]
        )
    messages[0] /= messages[0].sum()
    for i, e in enumerate(ids[-2:0:-1]):
        for value_end in states:
            messages[i+1][value_end] = np.einsum('i,i,i->', node_potentials[e], edge_potentials[e-1,value_end,:], messages[i])
        messages[i+1] /= messages[i+1].sum()
    return messages

def compute_marginals(y, transition_types, bins, mu, sigma, rhos):
    """
    Given data, bins, and parameters, computes marginal
    distributions.
    :param y: p length binary numpy array of emissions
    :param transition_types: p - 1 length array of transition 
    types. Its unique values should be k successive integers,
    zero-indexed.
    """
    node_potentials, edge_potentials = fetch_potentials(
        y=y, 
        transition_types=transition_types,
        bins=bins,
        mu=mu,
        sigma=sigma,
        rhos=rhos,
    )

    # Message passing
    fwd_message = compute_fwd_messages(
        node_potentials, edge_potentials
    )
    bkw_message = compute_bkw_messages(
        node_potentials, edge_potentials
    )

    # Summation to create marginals
    marginals = np.zeros_like(node_potentials)
    marginals[0] = bkw_message[-1]*node_potentials[0]
    for i in range(1,node_potentials.shape[0]-1):
        marginals[i] = fwd_message[i-1]*node_potentials[i]*bkw_message[-i-1]
    marginals[-1] = fwd_message[-1]*node_potentials[-1]
    
    # Normalize
    marginals /= marginals.sum(axis=-1, keepdims=True)
    return marginals


def sample_hmm(
    node_potentials,
    edge_potentials,
    bkw_messages
):

    # Figure out possible IDs and states
    ids = np.arange(node_potentials.shape[0])
    states = np.arange(node_potentials.shape[1])

    # Go up the tree!
    config = np.empty_like(ids)
    # 0 be its own case
    marginals = node_potentials[0]*bkw_messages[-1]
    config[0] = choices(states, weights=marginals/marginals.sum())[0]
    for e in ids[1:-1]:
        # Get (conditional) marginals
        marginals = edge_potentials[e-1,config[e-1],:]*node_potentials[e]*bkw_messages[-e-1]
        # Sample from the (conditional) marginal
        config[e] = choices(states, weights=marginals/marginals.sum())[0]
    # -1 be its own case
    marginals = edge_potentials[-1,config[-2],:]*node_potentials[-1] # The first -1 *is* right because edge_potentials is shorter.
    config[-1] = choices(states, weights=marginals/marginals.sum())[0]

    return config

def fetch_potentials(y, transition_types, bins, mu, sigma, rhos):
    """
    Given observed data, bins, and parameters,
    converts into node and edge potentials.
    :param y: p length binary numpy array of emissions
    :param transition_types: p - 1 length array of transition 
    types. Its unique values should be k successive integers,
    zero-indexed.
    """
    # Node potentials
    node_potentials = generate_node_potentials(y, bins, mu, sigma)
    transition_matrices = {
        rho: generate_transition_matrix(
            bins, rho, mu, sigma
        ) for rho in rhos
    }

    # Edge potentials
    ordered_rhos = rhos[transition_types]
    edge_potentials = np.stack([
        transition_matrices[rho] for rho in ordered_rhos
    ])
    return node_potentials, edge_potentials


def sample_conditional(
        y, 
        transition_types,
        bins,
        mu,
        sigma,
        rhos,
        n_sample=1,
        verbose=False
    ):
    node_potentials, edge_potentials = fetch_potentials(
        y=y, 
        transition_types=transition_types,
        bins=bins,
        mu=mu,
        sigma=sigma,
        rhos=rhos,
    )
    bkw_messages = compute_bkw_messages(
        node_potentials, edge_potentials
    )
    sample = []
    loop = trange(n_sample) if verbose else range(n_sample)
    for i in loop:
        sample.append(bins[sample_hmm(
                node_potentials=node_potentials,
                edge_potentials=edge_potentials, 
                bkw_messages=bkw_messages
        )])
    return np.array(sample) if n_sample > 1 else sample[0]