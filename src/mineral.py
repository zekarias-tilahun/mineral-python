import helper
import logging
import argparse
import sys
import numpy as np

from gensim.models import Word2Vec

logging.basicConfig(level=logging.ERROR)


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
