# Mineral
Implementation of the Mineral algorithm as described in the paper, 
[Mineral: Multi-modal Network Representation Learning](https://zekarias-tilahun.github.io/zack/publications/mineral-mod2018.pdf).
### Requirements!
  - gensim 0.13+
  - Numpy 1.14+
## Usage
#### Example usage
```sh
$ python src/mineral.py --net-file ../data/network.txt 
--emb-file ../data/network.txt
```

#### Input format
`network file`

```text
>Format
source target weight

>Example
0 1 0.2
0 2 0.4
1 2 0.7
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