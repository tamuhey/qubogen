# Examples of QUBOgen

## [Number Partitioning](https://en.wikipedia.org/wiki/Partition_problem)

How to partition a given multiset of numbers into two subsets S1, S2 such that |sum(S1) - sum(S2)| is as small as possible?

```python
import qubogen
from dwave_qbsolv import QBSolv
```

sample multiset

```python
s = np.array([3,1,1,2,2,1])
```

create qubo matrix with `qubogen.qubo_number_partition`

```python
qubogen.qubo_number_partition(s)
```

    array([[-21,   3,   3,   6,   6,   3],
           [  3,  -9,   1,   2,   2,   1],
           [  3,   1,  -9,   2,   2,   1],
           [  6,   2,   2, -16,   4,   2],
           [  6,   2,   2,   4, -16,   2],
           [  3,   1,   1,   2,   2,  -9]])

## [Graph Coloring](https://en.wikipedia.org/wiki/Graph_coloring)

- For a given graph, how to color the vertices such that no two adjacent vertices are of same color?

sample graph: builtin data structure `qubogen.Graph`

```python
edges = [[0,1],[0,4],[1,2],[1,3],[1,4],[2,3],[3,4]]
n_nodes
g = qubogen.Graph(edges=edges, n_nodes=n_nodes)
```

create qubo matrix with `qubogen.qubo_graph_coloring`

```python
q=qubogen.qubo_graph_coloring(g, n_color=3)
```

## Other combinatorial optimization problems

`qubogen` is also available for the below problems:

| problem                                      | method                   |
| -------------------------------------------- | ------------------------ |
| Max-Cut Problem                              | `qubogen.qubo_max_cut`   |  |
| Minimum Vertex Cover Problem (MVC)           | `qubogen.qubo_mvc`       |
| Weighted Minimum Vertex Cover Problem (WMVC) | `qubogen.qubo_wmvc`      |
| Set Packing Problem                          | `qubogen.qubo_set_pack`  |
| Max 2-Sat Problem                            | `qubogen.qubo_max2sat`   |
| Set Partitioning Problem (SPP)               | `qubogen.qubo_spp`       |
| General 0/1 Programming                      | `qubogen.qubo_general01` |
| Quadratic Assignment Problem (QAP)           | `qubogen.qubo_qap`       |
| Quadratic Knapsack Problem (QKP)             | `qubogen.qubo_qkp`       |
