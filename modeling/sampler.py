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

    # Figure out possible IDs and states
    ids = np.arange(node_potentials.shape[0])
    states = np.arange(node_potentials.shape[1])

    # Go up the tree!
    config = np.empty_like(ids)
    # 0 be its own case
    marginals = node_potentials[0]*bkw_messages[-1]
    config[0] = np.random.choice(states, p=marginals/marginals.sum())
    for e in ids[1:-1]:
        # Get (conditional) marginals
        marginals = edge_potentials[e-1,config[e-1],:]*node_potentials[e]*bkw_messages[-e-1]
        # Sample from the (conditional) marginal
        config[e] = np.random.choice(states, p=marginals/marginals.sum())
    # -1 be its own case
    marginals = edge_potentials[-1,config[-2],:]*node_potentials[-1] # The first -1 *is* right because edge_potentials is shorter.
    config[-1] = np.random.choice(states, p=marginals/marginals.sum())

    return config

def sample_conditional(df, bins, ro, mu, sigma, n_sample=1, verbose=False):
    node_potentials = generate_node_potentials(df, bins)
    edge_potentials = np.repeat(
        generate_transition_matrix(bins, ro, mu, sigma).reshape((1,bins.shape[0],bins.shape[0])),
        len(df)-1, axis=0
    )
    bkw_messages = compute_bkw_messages(node_potentials, edge_potentials)
    sample = []
    loop = trange(n_sample) if verbose else range(n_sample)
    for i in loop:
        sample.append(bins[sample_hmm(node_potentials, edge_potentials, bkw_messages)])
    return np.array(sample) if n_sample > 1 else sample[0]

if __name__ == '__main__':
    df = process_data()
    sample = sample_conditional(df.head(50), np.round(np.arange(-2.5,2.6,.1), 3), 0.5, 0, 1, n_sample=5, verbose=True)
    print(sample.shape)