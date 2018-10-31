import numpy as np
import logging
import datetime as dt

logging.basicConfig(level=logging.INFO)


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
