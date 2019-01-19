from sklearn.preprocessing import normalize
from scipy.sparse import csr_matrix
import networkx as nx
import numpy as np


def read_network(path, directed=False, input_format='edgelist', sep='\t'):
    """
    Reads a graph from a path
    :param path: The path
    :param directed: An flag to indicate if the graph is directed or not
    :param input_format: The format of the file, possible values are
                         (edgelist - Default | adjlist | mattxt | matnpy)
    :param sep:
    :return:
    """
    print('INFO: Reading network file from {} stored as {} format'.format(
        path, input_format))
    create_using = nx.DiGraph() if directed else nx.Graph()
    if input_format == 'edgelist':
        network = nx.read_edgelist(
            path, nodetype=int, create_using=create_using)
        adj_mat = nx.to_scipy_sparse_matrix(network, sorted(network.nodes()))
    elif input_format == 'adjlist':
        network = nx.read_adjlist(
            path, nodetype=int, create_using=create_using)
        adj_mat = nx.to_scipy_sparse_matrix(network, sorted(network.nodes()))
    elif input_format == 'mattxt':
        adj_mat = csr_matrix(np.loadtxt(path))
    else:
        adj_mat = csr_matrix(np.load(path))

    norm_adj_mat = normalize(adj_mat, norm='l1')
    network = nx.from_scipy_sparse_matrix(A=norm_adj_mat, create_using=create_using)
    print('\n\tNumber of nodes: {}\n\tNumber of edges: {}'.format(
        network.number_of_nodes(), network.number_of_edges()))
    return network


def build_feature_matrix(path, num_nodes, input_format='adjlist'):
    """
    Builds nodes feature matrix from an attribute file
    :param path: A path to nodes attribute file
    :param num_nodes: The number of nodes
    :param input_format: The file format, possible values are
            (adjlist - Default | edgelist | mattxt | matnpy), for large
            files use adjlist or edgelist
    :return: Numpy array or Scipy sparse matrix
    """
    print('INFO: Reading attribute file from {} stored as {} format'.format(
        path, input_format))
    if input_format == 'adjlist':
        reader = nx.read_adjlist
    elif input_format == 'edgelist':
        reader = nx.read_edgelist
    elif input_format == 'mattxt':
        mat = np.loadtxt(path)
        print('INFO: Number of features: {}'.format(mat.shape[1]))
        return mat
    else:
        mat = np.load(path)
        print('INFO: Number of features: {}'.format(mat.shape[1]))
        return mat

    attributes = reader(
        path, create_using=nx.DiGraph(), nodetype=int)
    attribute_mat = nx.to_scipy_sparse_matrix(
        attributes, nodelist=sorted(attributes.nodes()))
    feature_mat = attribute_mat[:num_nodes, :num_nodes]
    return feature_mat


def read_cascades(cas_file, min_threshold, max_threshold):
    """
    Reading cascades from a file path

    :param cas_file:
    :param min_threshold:
    :param max_threshold:
    :return:
    """
    print('INFO: Reading existing cascades from {} ...'.format(cas_file))
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


def read_embedding(path, existing_emb=None, sep=None):
    """
    Read nodes embedding (representation) from a file
    :param path:
    :param existing_emb
    :param sep:
    :return:
    """
    print('INFO: Reading embedding from {}'.format(path))
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


def build_cascade_graph(cascades, num_nodes):
    """
    Construct a cascade graph G_c = (V_c, E_c), Where V_c contains the set of
    nodes V from the original graph G, and additional set of nodes U representing
    cascades. V_c = Union(V, U) and each edge (v, u) is an edge between a nodes
    u and v, which are members of V and U respectively.
    :param cascades: The cascades
    :param num_nodes: The number of nodes |V_c|, it is used as the starting point
                      for generating cascade ids.
    :return: A networkx graph
    """
    cid = num_nodes
    edges = []
    for c in cascades:
        edges += [(node, cid) for node in c]
        cid += 1
    g = nx.from_edgelist(edges)
    return g


def save_embedding(path, model):
    print('INFO: Saving learned embeddings to {} using word2vec format'.format(path))
    model.wv.save_word2vec_format(path)


def save_cascades(path, cascades):
    with open(path, 'w') as f:
        for c in cascades:
            f.write('{}\n'.format(' '.join(str(n) for n in c)))
