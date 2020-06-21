import numpy as np
from tqdm import *
import random, sys
from functools import reduce
from operator import mul

from .util import *
from .processing import *

# This may not scale to lots more points but is working for now
sys.setrecursionlimit(10000)

def neighbor_builder(hmm_length):
    def neighbors(node):
        if node > 0 and node < hmm_length-1: return (node-1,node+1)
        if node > 0: return (node-1,)
        return (node+1,)
    return neighbors

def hmm_sampler(node_potentials, edge_potentials):
    # Make sure shapes, are, like, reasonable
    assert(node_potentials.shape[1] == edge_potentials.shape[1])
    assert(edge_potentials.shape[1] == edge_potentials.shape[2])

    # Don't damage them, please
    node_potentials = node_potentials.copy()

    # Figure out possible IDs and states
    ids = np.arange(node_potentials.shape[0])
    states = np.arange(node_potentials.shape[1])

    neighbors = neighbor_builder(node_potentials.shape[0])

    get_edge_potential = lambda s, e, v_s, v_e: edge_potentials[s,v_s,v_e] if s < e else edge_potentials[e,v_e,v_s]

    messages = np.zeros(ids.shape+(2,)+states.shape, dtype=np.float32)-1

    # Recursively compute messages
    def compute_message(start, end, end_value):
        if messages[start][end][end_value] != -1.: return messages[start][end][end_value]
        # Now we're going to compute ALL of the messages, since it makes normalization easier.
        s_sum = 0
        for value_end in states:
            messages[start][end][value_end] = sum(
                reduce(mul, [node_potentials[start][value_i], get_edge_potential(start,end,value_i,value_end)]+[
                    compute_message(neighbor, start, value_i)
                    for neighbor in neighbors(start) if neighbor != end
                ]) for value_i in states
            )
            s_sum += messages[start][end][value_end]
        messages[start][end] /= s_sum
        return messages[start][end][end_value]
    # To avoid ridiculous recursion depth, run the backward pass first in a loop to precompute everything
    for e in reversed(ids[:-1]):
        for v in states:
            compute_message(e+1,e,v)
    # Go up  the tree!
    config = np.zeros_like(ids)
    for e in ids:
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

def sample_conditional(df, bins, ro, mu, sigma, n_sample=1, verbose=False):
    node_potentials = generate_node_potentials(df, bins)
    edge_potentials = np.repeat(
        generate_transition_matrix(bins, ro, mu, sigma).reshape((1,bins.shape[0],bins.shape[0])),
        len(df)-1, axis=0
    )
    sample = []
    loop = trange(n_sample) if verbose else range(n_sample)
    for i in loop:
        sample.append(bins[hmm_sampler(node_potentials, edge_potentials)])
    return np.array(sample) if n_sample > 1 else sample[0]

if __name__ == '__main__':
    df = process_data()
    sample = sample_conditional(df.head(50), np.round(np.arange(-2.5,2.6,.1), 3), 0.5, 0, 1, n_sample=5, verbose=True)
    print(sample.shape)