---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Basics of NetworkX

The best-known package for working with networks is NetworkX.

```{code-cell} ipython3
import networkx as nx
```

One way to create a graph is from a list of edges.

```{code-cell} ipython3
star = nx.Graph( [(1,2),(1,3),(1,4),(1,5),(1,6)] )
nx.draw(star)
```

Another way to create a graph is to give the start and end nodes of the edges as columns in a data frame.

```{code-cell} ipython3
import pandas as pd
df = pd.DataFrame( {'from':[1,2,3,4,5,6],'to':[2,3,4,5,6,1]} )
H = nx.from_pandas_edgelist(df,'from','to')
nx.draw(H)
```

There are functions that generate different well-studied types of graphs. A **complete graph** is one that has every possible edge.

```{code-cell} ipython3
K5 = nx.complete_graph(5)
print("5 nodes,",nx.number_of_edges(K5),"edges")
nx.draw(K5)
```

A **lattice graph** has a regular structure, like graph paper.

```{code-cell} ipython3
lat = nx.grid_graph((6,4))
print(lat.number_of_nodes(),"nodes,",lat.number_of_edges(),"edges")
nx.draw(lat,node_size=100)
```

An **Erdős-Rényi graph** includes each individual possible edge with a fixed probability $p$. When one refers to a "random graph" without any additional context, this is the type that is meant. 

```{code-cell} ipython3
er = nx.erdos_renyi_graph(60,.08,seed=1)
print(er.number_of_nodes(),"nodes,",er.number_of_edges(),"edges")
nx.draw_circular(er,node_size=40)
```

A **Watts–Strogatz graph** is randomized but prefers links to a small group of neighbors over distant nodes.

```{code-cell} ipython3
ws = nx.watts_strogatz_graph(50,5,.2,seed=4)
nx.draw(ws,node_size=100)
```

There are different ways to draw a particular graph in the plane, as determined by the positions of the nodes. The default is to imagine that the edges are springs pulling on the nodes. But there are alternatives that may be useful at times.

```{code-cell} ipython3
nx.draw_circular(K5)
```

There are many ways to read graphs from (and write them to) files. Here is a friend network among Twitch users. The file has a pair of nodes representing one edge on each line.

```{code-cell} ipython3
twitch = nx.read_edgelist("musae_edges.csv",delimiter=',',nodetype=int)
```

```{code-cell} ipython3
twitch.number_of_nodes(),twitch.number_of_edges()
```

## Ego graphs

We can extract for any node of a graph its **ego graph**, which is the subset of nodes it is connected to, along with the associated edges.

```{code-cell} ipython3
ego = nx.ego_graph(twitch,400)
```

```{code-cell} ipython3
nx.draw(ego,with_labels=True,node_size=800,node_color="yellow")
```

Notice that the nodes of the ego network have the same labels as they did in the original graph that it was taken from.

We can widen the ego graph to include the ego graphs of all the neighbors:

```{code-cell} ipython3
big_ego = nx.ego_graph(twitch,400,radius=2)
big_ego.number_of_nodes()
```

```{code-cell} ipython3
pos = nx.spring_layout(big_ego,iterations=100)
nx.draw(big_ego,pos=pos,width=0.2,node_size=10,node_color="purple")
```

## Adjacency matrix

+++

Two nodes are said to be **adjacent** if there is an edge between them. Every graph can be associated with an **adjacency matrix**. Suppose the nodes are numbered from $0$ to $n-1$. The adjacency matrix is $n\times n$ and has a 1 at position $(i,j)$ if node $i$ and node $j$ are adjacent, and a 0 otherwise.

```{code-cell} ipython3
A = nx.adjacency_matrix(ego)
A
```

Observe that `A` is not stored in the format we have been used to. In a large network we would expect most of its entries to be zero, so it makes more sense to store it as a *sparse matrix*, where we keep track of only the nonzero entries:

```{code-cell} ipython3
print(A[:3,:])
```

We can easily convert `A` to a standard array, if it is not too large to fit in memory.

```{code-cell} ipython3
A.toarray()
```

In an undirected graph, we have $A_{ij}=A_{ji}$ everywhere, and we say that $A$ is *symmetric*. 

+++


## Connectedness

+++

We say that two nodes in a network are **connected** if there is a path of edges between them. If every pair of nodes in the network are connected, then we say the *graph* is connected.

```{code-cell} ipython3
nx.is_connected(twitch)
```

If a subset $C$ of nodes (with associated edges) forms a connected graph, and there are no links from nodes in $C$ to nodes outside of $C$, then $C$ is a **connected component**. A connected graph has one connected component; the other extreme is a graph with no edges, in which the number of connected components equals the number of nodes. 

```{code-cell} ipython3
ego.remove_edge(400,5079)
nx.draw_circular(ego,with_labels=True,node_size=800,node_color="yellow")
```

```{code-cell} ipython3
:tags: []

nx.number_connected_components(ego)
```

An **articulation node** is a node that, when removed (with its incident edges) from a graph, increases the number of connected components. Each articulation node is a "choke point" or hub for communication between otherwise disconnected subgraphs. The Twitch network has many such nodes.

```{code-cell} ipython3
ap = list(nx.articulation_points(twitch))
len(ap)
```

Removing an articulation node results in at least two connected components, but there might be more.

```{code-cell} ipython3
t2 = twitch.copy()
t2.remove_node(ap[0])
nx.number_connected_components(t2)
```
