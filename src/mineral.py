import helper
import logging
import argparse
import sys
import numpy as np
import datetime as dt

from gensim.models import Word2Vec

logging.basicConfig(level=logging.ERROR)


def alias_setup(dist):
    """
    Set up the alias and probability tables of Vose's Alias sampling method.

    :param dist:
    :return: (alias table, probability table, node look up table)
    """

    logging.debug('{} Initializing the alias and probability tables'.format(dt.datetime.now()))
    k = len(dist)
    probability_table = np.zeros(k)
    alias_table = np.zeros(k, dtype=np.int)
    # Sort the data into the outcomes with probabilities
    #  that are larger and smaller than 1/K.
    smaller = []
    larger = []
    index = 0
    index_to_node = {}
    for node, prob in dist.items():
        index_to_node[index] = node
        probability_table[index] = k * prob
        if probability_table[index] < 1.0:
            smaller.append(index)
        else:
            larger.append(index)
        index += 1

    # Loop though and create little binary mixtures that
    # appropriately allocate the larger outcomes over the
    #  overall uniform mixture.
    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        alias_table[small] = large
        probability_table[large] -= 1.0 - probability_table[small]

        if probability_table[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return alias_table, probability_table, index_to_node


def alias_draw(alias_table, probability_table):
    """
    Draw samples from either the alias or probability table

    :param alias_table:
    :param probability_table:
    :return:
    """
    K = len(alias_table)
    # Draw from the overall uniform mixture.
    index = int(np.floor(np.random.uniform() * K))
    # Draw from the binary mixture, either keeping the
    # small one, or choosing the associated larger one.
    if np.random.uniform() < probability_table[index]:
        return index
    else:
        return alias_table[index]


def simulate_diffusion(network, root, r, h):
    """
    :param network: 
    :param root: A root from which the diffusion process starts from
    :param r: parameter for the number of times a diffusion is to be simulated from the root
    :param h: The maximum number of nodes to be infected after the simulation
    :return: 
    """
    cascades = []
    for i in range(r):
        size_limit_reached = False
        cascade = [[root]]
        infected = {root}
        t = 1
        while not size_limit_reached:
            currently_infected_nodes = []
            previously_infected_nodes = cascade[t - 1]

            for node in previously_infected_nodes:
                if node in network:
                    for nbr in network[node]:
                        if nbr not in infected:
                            weight = network[node][nbr]
                            r = np.random.random()
                            if r < weight:
                                currently_infected_nodes.append(nbr)
                                infected.add(nbr)
                                if len(infected) >= h:
                                    size_limit_reached = True
                                    break

                if size_limit_reached:
                    break
            if len(currently_infected_nodes) == 0:
                break
            cascade.append(currently_infected_nodes)
            logging.debug('Current Cascades {}'.format(cascade))
            t += 1
        cascades.append([node for current_infection in cascade for node in current_infection])
    return cascades


def mineral_setup(network):
    """
    Builds node configuration tables by maintaining a three table structures to facilitate
    constant time neighbor sampling. Thus for a given node u, the node configuration
    will look like
          node_config[u] = (alias_table, probability_table, index_to_node)
          alias_table and probability_table are data structures used for the sampling
          according to Vose's alias sampling method

          index_to_node is a look up table and associates an index (automatically generated)
          to a unique node.

    :param network:
    :return:
    """
    logging.info('Setting up the network to facilitate constant time edge sampling')
    nodes_config = {}
    for src in network:
        norm_const = np.sum(network[src].values())
        for dst in network[src]:
            network[src][dst] /= norm_const

        nodes_config[src] = alias_setup(network[src])
    return nodes_config


def mineral_cascades(network, r, h):
    """
    Generate cascades using a number of simulation of truncated diffusion processes from
    each node
    
    :param network: 
    :param r: 
    :param h: 
    :return: 
    """
    logging.info('Simulating diffusion ...')
    nodes = network.keys()
    cascades = []
    progress = 0
    for root in nodes:
        progress += 1
        sys.stdout.write('\r{}/{} nodes have been processed'.format(progress, len(nodes)))
        sys.stdout.flush()
        cascades += simulate_diffusion(network, root, r, h)

    return cascades


def learn_embedding(walks, d, window, epoch, workers=8):
    return Word2Vec(walks, size=d, window=window, min_count=0,
                    iter=epoch, sg=1, workers=workers)


def display_args(args):
    logging.info('Input arguments')
    for arg in vars(args):
        logging.info('{}: {}'.format(arg, getattr(args, arg)))


def parse_args():
    parser = argparse.ArgumentParser(description="Runs diffusion simulator")
    parser.add_argument('--net-file', '-n', default='', help='Path to network file')
    parser.add_argument('--cas-file', '-c', default='', help='Path to existing cascade file')
    parser.add_argument('--emb-file', '-e', default='', help='Path to the embedding output file')
    parser.add_argument('--directed', dest='directed', action='store_true')
    parser.add_argument('--undirected', dest='directed', action='store_false')
    parser.set_defaults(directed=True)
    parser.add_argument('--weighted', dest='weighted', action='store_true')
    parser.add_argument('--unweighted', dest='weighted', action='store_false')
    parser.set_defaults(weighted=True)
    parser.add_argument('--min-threshold', '-mn', type=int, default=10, help='Minimum cascade length to consider')
    parser.add_argument('--max-threshold', '-mx', type=int, default=500, help='Maximum cascade length to consider')
    parser.add_argument('--dim', '-d', type=int, default=100, help='Size of the representation')
    parser.add_argument('--window', type=int, default=10, help='Window size')
    parser.add_argument('--iter', type=int, default=20, help='Number of epochs')
    parser.add_argument('--r', type=int, default=10, help='Number of diffusion processes to simulate from a node')
    parser.add_argument('--h', type=int, default=60, help='Maximum number of nodes to infect in a single simulation')
    return parser.parse_args()


def main():
    args = parse_args()
    display_args(args)
    network = helper.read_network(args.net_file, directed=args.directed)
    cascades = mineral_cascades(network, r=args.r, h=args.h)
    cascades = [map(str, cascade) for cascade in cascades]
    if len(cascades) > 0:
        if args.cas_file != '':
            logging.info('With cascades')
            existing_cascades = helper.read_cascades(args.cas_file, args.min_threshold, args.max_threshold)
            existing_cascades = [map(str, cascade) for cascade in existing_cascades]
            cascades += existing_cascades
            model = learn_embedding(cascades, d=args.dim, window=args.window, epoch=args.iter)
        else:
            logging.info('Without cascades')
            model = learn_embedding(cascades, d=args.dim, window=args.window, epoch=args.iter)
        model.save_word2vec_format(args.emb_file)
    else:
        logging.error('The length of the cascades is zero, nothing to train on')


if __name__ == '__main__':
    main()
