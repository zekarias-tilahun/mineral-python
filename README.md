# Mineral (0.0.2)
Implementation of the Mineral algorithm as described in the paper, 
[Mineral: Multi-modal Network Representation Learning](https://zekarias-tilahun.github.io/zack/publications/mineral-mod2018.pdf).
This version includes a minor change for cascade sampling, 
if observed cascades are provided.
To enable this functionality just activate the `--sample` flag.
Without this flag, it is exactly the first version Mineral 0.0.1.
### Requirements!
  - gensim 0.13+
  - Numpy 1.14+
  - networkx  2.0+
  
## Usage
#### Example usage
```sh
$ python src/mineral.py --net-file ../data/network.txt --emb-file ../data/cascades.txt
```

#### Input Files
##### Graph inputs
`net-file` and `att-file`

Four kinds of formats are supported for both files, 
which are `adjlist`, `edgelist`, `mattxt`, and `matnpy`

>`adjlist`

```text
Format
node (list of neighbors | list of attributes)

Example
0 1 2 3
1 0 3
2 0
3 0 1
```

>`edgelist`

```text
Format
node (neighbor | attribute)

Example
0 1
0 2
0 3
1 0
1 3
2 0
3 0
3 1
```

>`mattxt` or `matnpy`

```text
Format
A matrix W of n x m dimensions

n is the number of nodes 
m is equal to n if W is an adjacency matrix of the graph
m is the number of attributes if W is an attribute matrix.
W can be a *.txt text or *.npy numpy binary file.
```


`cas-file`

```text
Format:
(list of nodes)

Example:
2 0 1
1 0
0 2 1
```

### Available command line options

`--net-file:`
Path to a network file. Default is ../data/network.txt

`--net-format:`
Network file format. Possible values are 
`edgelist`, `adjlist`, `mattxt`, and `matnpy` .
Default is `edgelist`

`--att-file` A path to nodes attribute file. Default is ../data/attributes.txt

`--att-format:` Attribute file format. Possible values are 
`edgelist`, `adjlist`, `mattxt`, and `matnpy`. 
Default is `adjlist`


`--cas-file:` A path to cascades file. Default is ../data/cascades.txt

`--sim-file:` A path to save the simulated cascades. Default is '../data/simulated_cascades.txt'

`--emb-file:` A path to save the embedding files. Default is ../data/network.emb

`--directed:` A flag to indicate the input graph is directed. Default is False

`--undirected:` A flag to indicate the input graph is undirected. Default is True

`--weighted:` A flag to indicate the input graph is weighted. Default is False

`--unweighted:` A flag to indicate the input graph is unweighted. Default is True

`--min-threshold:` A threshold for the min length of observed cascades

`--max-threshold:` A threshold for the max length of observed cascades

`--dim:` An embedding dimension. Default is 128.

`--window:` A window size for the SkipGram model. Default is 10

`--iter:` The number of epochs. Default is 20

`--r:` The number of cascades to simulate from each node. Default is 10

`--h:` The max length of each simulated cascade. Default is 80

Citing
------
If you find Mineral relevant to your research, we kindly ask you to cite the paper:

```
@inproceedings{MOD17,
    booktitle="Proc. of the 3rd International Conference on Machine Learning, Optimization and Big Data",
    series = "MOD'17",
    author = {Kefato, Zekarias T. and Sheikh, Nasrullah and Montresor, Alberto},
    title = "Mineral: Multi-modal Network Representation Learning",
    month = sep,
    year = 2017,
    type = {CONFERENCE},	
    publisher = {Springer},
}
```
