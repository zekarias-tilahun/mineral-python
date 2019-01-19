from sklearn.preprocessing import normalize
from gensim.models import Word2Vec

from mineral_helper import *

import networkx as nx
import numpy as np

import argparse


def sample_cascade(cascade_graph, root, h, num_nodes):
    """
    Uniformly samples a cascade from a cascade graph starting at a given root
    node. The length of the cascade is bounded by an upper value
    :param cascade_graph: The cascade graph
    :param root: The root node to sample a cascade from
    :param h: The cascade length
    :param num_nodes: The number of nodes used to control nodes to add to the
                      sampled cascade
    :return: list, A cascade [sequence of nodes]
    """
    cascade = [root]
    next_nodes = [root]
    while len(next_nodes) < h:
        current_node = next_nodes[-1]
        if current_node in cascade_graph:
            neighbors = cascade_graph[current_node]
            next_node = np.random.choice(neighbors)
            next_nodes.append(next_node)
            if next_node < num_nodes:
                cascade.append(next_node)
        else:
            break
    return cascade


def sample_cascades(cascade_graph, num_nodes, r, h):
    """
    Samples cascades from observed cascades modeled as a cascade graph

    :param cascade_graph: The cascade graph
    :param num_nodes: The number of nodes in the actual graph
    :param r: The number of cascades to sample from each node
    :param h: The length of a sampled cascade
    :return: list of lists: A list containing sampled cascades
    """
    cascades = []
    nodes = cascade_graph.nodes()
    for i in range(r):
        print('PROGRESS: {}/{}'.format(i + 1, r))
        np.random.shuffle(nodes)
        for root in nodes:
            cascade = sample_cascade(
                cascade_graph, root=root, h=h, num_nodes=num_nodes)
            cascades.append(cascade)
    return cascades


def compute_similarity(network, feature_matrix):
    """
    Builds a weighted graph, where each edge (u, v, sim) is constructed by
    computing the Jaccard similarity (sim) of the attributes of nodes u and v.
    For the sake efficiency, a vectorized implementation is used
    :param network: The network
    :param feature_matrix: Attribute information encoded as a feature matrix
            A row is a node and each column is associated with the attributes.
            feature_matrix[i, j] = 1 if node i has attribute j otherwise 0.
    :return:
    """
    print('INFO: Computing similarity between incident nodes of '
          'each edge in the graph')
    edges = network.edges()
    adj_mat = nx.to_scipy_sparse_matrix(network, sorted(network.nodes()))
    sources, targets = list(zip(*edges))
    sources, targets = list(sources), list(targets)

    '''
        Computing common attributes. The matrix common_attributes
        has the same number of rows as the number of edges due to
        feature_matrix[sources] + feature_matrix[targets].
        If common_attributes[(i, j), k] = 0, neither i nor j has the 
        k-th attribute, if common_attributes[(i, j), k] = 1, either of
        them has the k-th attribute, if common_attributes[(i, j), k] = 2
        both of them has the k-th attribute.
    '''
    common_attributes = feature_matrix[sources] + feature_matrix[targets]
    num_common_attributes = np.apply_along_axis(
        lambda arr: arr[arr == 2].size, axis=1, arr=common_attributes)

    # Computing combined attributes
    num_combined_attributes = np.apply_along_axis(
        lambda arr: arr[arr > 0].size, axis=1, arr=common_attributes)

    # Jaccard similarity
    similarities = num_common_attributes / num_combined_attributes

    # Construct a weighted graph based on the similarities
    adj_mat[sources, targets] = similarities
    norm_adj_mat = normalize(adj_mat, norm='l2')
    return nx.from_scipy_sparse_matrix(norm_adj_mat)


def simulate_diffusion(network, root, h):
    """
    Simulates an information diffusion processes in-order to sample cascades

    :param network: The network
    :param root: The root node from which cascades will be sampled
    :param h: The maximum length of a cascade sample
    :return: list: A cascade (A sequence of nodes)
    """
    infections = {0: {root}}
    cascade = [root]
    current_time_step = 1
    while len(cascade) < h:
        previously_infected_nodes = infections[current_time_step - 1]
        infections[current_time_step] = set()
        for node in previously_infected_nodes:
            if node in network:
                for nbr in network[node]:
                    if nbr not in infections[current_time_step]:
                        w = network[node][nbr]['weight']
                        if np.random.random() < w:
                            infections[current_time_step].add(nbr)
                            cascade.append(nbr)
                            if len(cascade) >= h:
                                return cascade
        if len(infections[current_time_step]) == 0:
            break
        current_time_step += 1
    return cascade


def simulate_diffusion_events(network, r, h):
    """
    Samples cascades using a number of simulation of truncated diffusion processes from
    each node
    
    :param network: 
    :param r: 
    :param h: 
    :return: 
    """
    print('INFO: Simulating diffusion ...')
    nodes = list(network.nodes())
    cascades = []
    for i in range(r):
        np.random.shuffle(nodes)
        for root in nodes:
            cascade = simulate_diffusion(network, root, h)
            cascades.append(cascade)
        print('PROGRESS: {}/{}'.format(i + 1, r))

    return cascades


def embed(walks, d, window, epoch, workers=8):
    model = Word2Vec(walks, size=d, window=window, min_count=0, iter=epoch, sg=1, workers=workers)
    return model


def display_args(args):
    print('INFO: Input arguments')
    for arg in vars(args):
        print('INFO: {}: {}'.format(arg, getattr(args, arg)))


def parse_args():
    parser = argparse.ArgumentParser(description="Runs the python implementation of mineral")
    parser.add_argument('--net-file', default='../data/cora/network.txt', help='Path to network file')
    parser.add_argument('--net-format', default='edgelist',
                        help='Graph file format, possible values are (edgelist, adjlist).'
                             'Default is edgelist')
    parser.add_argument('--att-file', default='../data/cora/attributes.txt', help='Path to attributes file')
    parser.add_argument('--att-format', default='mattxt',
                        help='Similar to graph file format. Default is mattxt')
    parser.add_argument('--cas-file', default='',
                        help='Path to observed cascades file')
    parser.add_argument('--sim-file', default='../data/cora/simulated_cascades.txt', help='Path to simulated cascade file')
    parser.add_argument('--emb-file', default='../data/cora/network.emb', help='Path to the embedding output file')
    parser.add_argument('--sample', dest='sample', action='store_true',
                        help="An indicator whether to sample from observed cascades."
                             "Valid when observed cascades are provided. "
                             "Default is False")
    parser.set_defaults(sample=False)
    parser.add_argument('--directed', dest='directed', action='store_true')
    parser.add_argument('--undirected', dest='directed', action='store_false')
    parser.set_defaults(directed=False)
    parser.add_argument('--weighted', dest='weighted', action='store_true')
    parser.add_argument('--unweighted', dest='weighted', action='store_false')
    parser.set_defaults(weighted=False)
    parser.add_argument('--min-threshold', type=int, default=5, help='Minimum cascade length to consider')
    parser.add_argument('--max-threshold', type=int, default=500, help='Maximum cascade length to consider')
    parser.add_argument('--dim', type=int, default=128, help='Size of the representation')
    parser.add_argument('--window', type=int, default=10, help='Window size')
    parser.add_argument('--iter', type=int, default=20, help='Number of epochs')
    parser.add_argument('--r', type=int, default=10,
                        help='Number of diffusion processes to simulate from a node')
    parser.add_argument('--h', type=int, default=60,
                        help='Maximum number of nodes to infect in a single simulation.'
                             'Default is 60')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel jobs. Default is 8')
    return parser.parse_args()


def main():
    args = parse_args()
    display_args(args)
    network = read_network(args.net_file, directed=args.directed)
    num_nodes = network.number_of_nodes()
    if args.att_file.strip() != '':
        feature_matrix = build_feature_matrix(
            args.att_file, input_format=args.att_format, num_nodes=num_nodes)
        network = compute_similarity(
            network=network, feature_matrix=feature_matrix)
        print('INFO: Attribute information is used to build a weighted graph')
    cascades = simulate_diffusion_events(network, r=args.r, h=args.h)
    if args.sim_file != '':
        save_cascades(args.sim_file, cascades)

    if len(cascades) > 0:
        if args.cas_file != '':
            observed_cascades = read_cascades(args.cas_file, args.min_threshold, args.max_threshold)
            if args.sample:
                cascade_graph = build_cascade_graph(
                    cascades=observed_cascades, num_nodes=num_nodes)
                sampled_cascades = sample_cascades(
                    cascade_graph=cascade_graph, num_nodes=num_nodes, r=args.r, h=args.h)
                cascades += sampled_cascades
            else:
                cascades += observed_cascades
            print('INFO: Learning with observed cascades')
        else:
            print('INFO:Learning without observed cascades')

        cascades = [list(map(str, cascade)) for cascade in cascades]
        model = embed(cascades, d=args.dim, window=args.window, epoch=args.iter, workers=args.workers)
        save_embedding(args.emb_file, model)
    else:
        raise ValueError('The length of the cascades is zero, nothing to train on')


if __name__ == '__main__':
    main()
