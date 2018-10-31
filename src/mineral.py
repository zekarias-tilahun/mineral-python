import logging
import argparse
import sys
import numpy as np

from gensim.models import Word2Vec

logging.basicConfig(level=logging.ERROR)


def read_embedding(path, existing_emb=None, sep=None):
    """
    Read nodes embedding (representation) from a file
    :param path:
    :param existing_emb
    :param sep:
    :return:
    """
    logging.info('{}:Reading embedding from {}'.format(dt.datetime.now(), path))
    nodes_embedding = {}
    with open(path) as f:
        for line in f:
            node_embedding = line.strip().split() if sep is None else line.strip().split(sep)
            if len(node_embedding) > 2:
                node = int(node_embedding[0])
                if existing_emb is not None and node in existing_emb:
                    embedding = np.array([float(num) for num in node_embedding[1:]])
                    nodes_embedding[node] = embedding
                elif existing_emb is None:
                    embedding = np.array([float(num) for num in node_embedding[1:]])
                    nodes_embedding[node] = embedding
    return nodes_embedding


def normalize(network):
    logging.info('Normalizing edge weights')
    for src in network:
        norm = 0
        for dst in network[src]:
            norm += network[src][dst] ** 2

        for dst in network[src]:
            network[src][dst] /= np.sqrt(norm)


def add_edge(g, src, dst, w, directed=True):

    if src not in g:
        g[src] = {dst: w}
    else:
        g[src][dst] = w
    if not directed:
        add_edge(g, dst, src, True)


def read_network(path, weighted=True, directed=False, sep=None):
    """
    Reads ground truth network from a file

    :param path:
    :param sep:
    :param weighted
    :param directed
    :return:
    """
    logging.info('{}:Reading network from {}'.format(dt.datetime.now(), path))
    network = {}
    with open(path) as f:
        for line in f:
            edge = line.strip().split() if sep is None else line.strip().split(sep)
            src, dst = int(edge[0]), int(edge[1])
            weight = float(edge[2]) if weighted else 1.0
            add_edge(network, src, dst, weight, directed)

    normalize(network)
    return network


def read_cascades(cas_file, min_threshold, max_threshold):
    logging.info('Reading existing cascades from {} ...'.format(cas_file))
    cascades = []
    with open(cas_file) as f:
        for line in f:
            cascade = []
            cas = line.strip().split()
            if min_threshold < len(cas) < max_threshold:
                for node in cas:
                    cascade.append(int(node))
                cascades.append(cas)
    return cascades


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


def embed(walks, d, window, epoch, workers=8):
    return Word2Vec(walks, size=d, window=window, min_count=0,
                    iter=epoch, sg=1, workers=workers)


def display_args(args):
    logging.info('Input arguments')
    for arg in vars(args):
        logging.info('{}: {}'.format(arg, getattr(args, arg)))


def parse_args():
    parser = argparse.ArgumentParser(description="Runs diffusion simulator")
    parser.add_argument('--net-file', default='', help='Path to network file')
    parser.add_argument('--cas-file', default='', help='Path to existing cascade file')
    parser.add_argument('--emb-file', default='', help='Path to the embedding output file')
    parser.add_argument('--directed', dest='directed', action='store_true')
    parser.add_argument('--undirected', dest='directed', action='store_false')
    parser.set_defaults(directed=True)
    parser.add_argument('--weighted', dest='weighted', action='store_true')
    parser.add_argument('--unweighted', dest='weighted', action='store_false')
    parser.set_defaults(weighted=True)
    parser.add_argument('--min-threshold', type=int, default=10, help='Minimum cascade length to consider')
    parser.add_argument('--max-threshold', type=int, default=500, help='Maximum cascade length to consider')
    parser.add_argument('--dim', type=int, default=100, help='Size of the representation')
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
            model = embed(cascades, d=args.dim, window=args.window, epoch=args.iter)
        else:
            logging.info('Without cascades')
            model = embed(cascades, d=args.dim, window=args.window, epoch=args.iter)
        model.save_word2vec_format(args.emb_file)
    else:
        logging.error('The length of the cascades is zero, nothing to train on')


if __name__ == '__main__':
    main()
