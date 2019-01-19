# Mineral
Implementation of the Mineral algorithm as described in the paper, 
[Mineral: Multi-modal Network Representation Learning](https://zekarias-tilahun.github.io/zack/publications/mineral-mod2018.pdf).
### Requirements!
  - gensim 0.13+
  - Numpy 1.14+
## Usage
#### Example usage
```sh
$ python src/mineral.py --net-file ../data/network.txt --emb-file ../data/cascades.txt
```

#### Input format
`network file`

```text
Format
source target weight

Example
0 1 0.2
0 2 0.4
1 2 0.7
```

`attribute file`

```text
Format
node [list-of-attributes]

Example
0 a k f
1 b a k j d
2 j f d e
```

`cascade-file`

```text
Format:
node_1 node_2 node_3 ... node_m
node_1 node_2 node_3 ... node_n

Example:
2 0 1
1 0
0 2 1
```

### Available command line options

`--net-file:`
Path to a network file. Default is ../data/network.txt

`--att-file` A path to nodes attribute file. Default is ../data/attributes.txt

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
