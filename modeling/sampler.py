import numpy as np
from tqdm import *

from .util import *
from .processing import *


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
        mul_arr = np.concatenate((node_potentials[0], edge_potentials[0,:,value_end]), axis=-1)
        messages[0][value_end] = np.sum(np.prod(mul_arr, axis=0))
    for e in ids[1:-1]:
        for value_end in states:
            mul_arr = np.concatenate((node_potentials[e], edge_potentials[e,:,value_end], messages[e-1]), axis=-1)
            messages[e][value_end] = np.sum(np.prod(mul_arr, axis=0))
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
        mul_arr = np.concatenate((node_potentials[-1], edge_potentials[-2,value_end,:]), axis=-1)
        messages[0][value_end] = np.sum(np.prod(mul_arr, axis=0))
    for i, e in enumerate(ids[-2:0:-1]):
        for value_end in states:
            mul_arr = np.concatenate((node_potentials[e], edge_potentials[e-1,value_end,:], messages[i]), axis=-1)
            messages[i+1][value_end] = np.sum(np.prod(mul_arr, axis=0))
    return messages

def compute_marginals(node_potentials, edge_potentials):
    fwd_message = compute_fwd_messages(node_potentials, edge_potentials)
    bkw_message = compute_bkw_messages(node_potentials, edge_potentials)
    marginals = np.zeros_like(node_potentials)
    marginals[0] = bkw_message[-1]*node_potentials[0]
    for i in range(1,node_potentials.shape[0]-1):
        marginals[i] = fwd_message[i-1]*node_potentials[i]*bkw_message[-i-1]
    marginals[-1] = fwd_message[-1]*node_potentials[-1]
    return marginals / marginals.sum(axis=-1)


def sample_hmm(node_potentials, edge_potentials, bkw_messages):

    # Don't damage these, please
    node_potentials = node_potentials.copy()

    # Figure out possible IDs and states
    ids = np.arange(node_potentials.shape[0])
    states = np.arange(node_potentials.shape[1])

    # Go up the tree!
    config = np.empty_like(ids)
    marginals = bkw_messages[-1]*node_potentials[0]
    config[0] = np.random.choice(states, p=marginals)
    fwd_message, last_fwd_message = np.empty_like(states), np.ones_like(states)
    for e in ids[1:]:
        # Compute forward messages, using wraparound trick to avoid annoyingness
        for value_end in states:
            mul_arr = np.concatenate((node_potentials[e-1], edge_potentials[e-1,:,value_end], last_fwd_message), axis=-1)
            fwd_message[value_end] = np.sum(np.prod(mul_arr, axis=0))
            last_fwd_message = fwd_message.copy()
        # Get (conditional) marginals
        marginals = np.array([
            reduce(mul, [node_potentials[e][v_i]]+[compute_message(neighbor, e, v_i) for neighbor in neighbors(e)])
            for v_i in states
        ])
        marginals /= np.sum(marginals)
        # Sample from the (conditional) marginal
        config[e] = np.random.choice(states, p=marginals)
        node_potentials[e] = np.zeros_like(states)
        node_potentials[e][config[e]] = 1
    return config

def sample_conditional(df, bins, mu, sigma, rhos, n_sample=1, verbose=False):
    node_potentials = generate_node_potentials(df, bins)
    ordered_rhos = rhos[df['point_type']][1:]
    edge_potentials = np.array([
        generate_transition_matrix(
            bins, rho, mu, sigma
        ).reshape((1,bins.shape[0],bins.shape[0])) for rho in ordered_rhos
    ])
    sample = []
    loop = trange(n_sample) if verbose else range(n_sample)
    for i in loop:
        sample.append(bins[hmm_sampler(node_potentials, edge_potentials)])
    return np.array(sample) if n_sample > 1 else sample[0]