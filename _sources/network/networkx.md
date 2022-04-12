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

# Graphs

+++

In mathematics, a network is represented as a **graph**. A graph is a collection of **nodes** (also called *vertices*) and **edges** that connect pairs of nodes. A basic distinction in graph theory is between an **undirected graph**, in which the edge $(a,b)$ is identical to $(b,a)$, and a **directed graph** or **digraph**, in which $(a,b)$ and $(b,a)$ are different potential edges. In either type of graph, each edge might be labeled with a numerical value, which results in a **weighted graph**.

Undirected, unweighted graphs will give us plenty to handle, and we will not seek to go beyond them. We also will not consider graphs that allow a node to link to itself.

+++

## NetworkX

The best-known Python package for working with networks is NetworkX.

```{code-cell} ipython3
import networkx as nx
import seaborn as sns
```

One way to create a graph is from a list of edges.

```{code-cell} ipython3
star = nx.Graph( [(1,2),(1,3),(1,4),(1,5),(1,6)] )
nx.draw(star,with_labels=True,node_color="lightblue")
```

Another way to create a graph is to give the start and end nodes of the edges as columns in a data frame.

```{code-cell} ipython3
import pandas as pd
df = pd.DataFrame( {'from':[1,2,3,4,5,6],'to':[2,3,4,5,6,1]} )
print(df)
H = nx.from_pandas_edgelist(df,'from','to')
nx.draw(H,with_labels=True,node_color="lightblue")
```

There are functions that generate different well-studied types of graphs. For instance, the graph above is a **cycle graph** or *ring lattice*.

```{code-cell} ipython3
nx.draw(nx.cycle_graph(9))
```

A **complete graph** is one that has every possible edge.

```{code-cell} ipython3
K5 = nx.complete_graph(5)
print("5 nodes,",nx.number_of_edges(K5),"edges")
nx.draw(K5)
```

In a graph on $n$ nodes, there are 

$$
\binom{n}{2} = \frac{n!}{(n-2)!2!} = \frac{n(n-1)}{2}
$$

unique pairs of distinct nodes. Hence, that is the number of edges in the undirected complete graph on $n$ nodes.

+++

A **lattice graph** has a regular structure, like graph paper.

```{code-cell} ipython3
lat = nx.grid_graph((6,4))
print(lat.number_of_nodes(),"nodes,",lat.number_of_edges(),"edges")
nx.draw(lat,node_size=100)
```

In an $m\times n$ lattice graph, there are $m-1$ edges in one direction repeated $n$ times, plus $n-1$ edges in the other direction, repeated $m$ times. Thus there are 

$$
(m-1)n + (n-1)m = 2mn-(m+n)
$$

edges altogether.

+++ {"tags": []}

There are different ways to draw a particular graph in the plane, as determined by the positions of the nodes. The default is to imagine that the edges are springs pulling on the nodes. But there are alternatives that may be useful at times.

```{code-cell} ipython3
nx.draw_circular(lat)
```

+++ {"tags": []}

As you can see, it's not easy to tell how similar two graphs are by comparing their plots.

+++

## Adjacency matrix

+++

Two nodes are said to be **adjacent** if there is an edge between them. Every graph can be associated with an **adjacency matrix**. Suppose the nodes are numbered from $0$ to $n-1$. The adjacency matrix is $n\times n$ and has a 1 at position $(i,j)$ if node $i$ and node $j$ are adjacent, and a 0 otherwise.

```{code-cell} ipython3
A = nx.adjacency_matrix(star)
A
```

Observe that `A` is not stored in the format we have been used to. In a large network we would expect most of its entries to be zero, so it makes more sense to store it as a *sparse matrix*, where we keep track of only the nonzero entries:

```{code-cell} ipython3
print(A)
```

We can easily convert `A` to a standard array, if it is not too large to fit in memory.

```{code-cell} ipython3
A.toarray()
```

+++ {"tags": []}

In an undirected graph, we have $A_{ij}=A_{ji}$ everywhere, and we say that $A$ is *symmetric*.

+++ {"tags": []}

## Importing networks

There are many ways to read graphs from (and write them to) files. For example, here is a friend network among Twitch users. The file has a pair of nodes representing one edge on each line.

```{code-cell} ipython3
:tags: []

twitch = nx.read_edgelist("musae_edges.csv",delimiter=',',nodetype=int)
```

```{code-cell} ipython3
:tags: []

twitch.number_of_nodes(),twitch.number_of_edges()
```

This graph is difficult to draw in its entirety. We can zoom in on a subset by selecting a node and its **ego graph**, which includes its adjacent nodes along with all edges between these nodes.

```{code-cell} ipython3
ego = nx.ego_graph(twitch,400)
nx.draw(ego,with_labels=True,node_size=800,node_color="yellow")
```

+++ {"tags": []}

Notice that the nodes of the ego network have the same labels as they did in the original graph that it was taken from.

We can widen the ego graph to include the ego graphs of all the neighbors:

```{code-cell} ipython3
:tags: []

big_ego = nx.ego_graph(twitch,400,radius=2)
print(big_ego.number_of_nodes(),"nodes and",big_ego.number_of_edges(),"edges")
pos = nx.spring_layout(big_ego,iterations=60)
nx.draw(big_ego,pos=pos,width=0.2,node_size=10,node_color="purple")
```

The reason for the two-step process in making the plot above is that computing the node positions via springs takes a hidden computational iteration. By calling that iteration explicitly, we were able to stop if early and save time.

+++

## Degree and average degree

The **degree** of a node is the number of edges that have the node as an endpoint. Equivalently, it is the number of nodes in its ego graph, minus the original node itself. Tthe **average degree** of a graph is the mean of the degrees of all of its nodes. 

The `degree` property of a graph gives a list of all nodes with their degrees.

```{code-cell} ipython3
ego.degree
```

The result here can be a bit awkward to work with; it's actually a *generator* of a list, rather than the list itself. (This "lazy" attitude is useful when dealing with very large networks.) So, for instance, we can collect the list as ordered tuples using a comprehension:

```{code-cell} ipython3
[d for d in ego.degree]
```

To compute the average degree, we can do a sum over the generator instead.

```{code-cell} ipython3
def average_degree(g):
    return sum(d[1] for d in g.degree)/g.number_of_nodes()

print("average degree of K5 complete graph:",average_degree(K5))
print("average degree of Twitch network:",average_degree(twitch))
```

There's a much easier way to compute this quantity, however. If we sum the degrees of all the nodes in a graph, we must get twice the number of edges in the graph.

```{code-cell} ipython3
def average_degree(g):
    return 2*g.number_of_edges()/g.number_of_nodes()

print("average degree of Twitch network:",average_degree(twitch))
```

## Random graphs

One way of understanding a real-world network is by comparing it to ones that are constructed randomly, but according to relatively simple rules. The idea is that if the real network behaves similarly to members of some random family, then perhaps it is constructed according to similar principles.

+++

An **Erdős-Rényi graph** (ER graph) includes each individual possible edge with a fixed probability $p$. That is, if you have a weighted coin that comes up heads (100p)% of the time, then you toss the coin for each possible pair of vertices and include their edge if it is heads.

```{code-cell} ipython3
n,p = 50,0.08
ER = nx.erdos_renyi_graph(n,p,seed=2)
print(ER.number_of_nodes(),"nodes,",ER.number_of_edges(),"edges")
nx.draw_circular(ER,node_size=50,edge_color="gray")
```

Since there are $\binom{n}{2}$ unique pairs among $n$ nodes, the mean number of edges in an ER graph is 

$$
p\binom{n}{2} = \frac{pn(n-1)}{2}.
$$

This fact is usually stated in terms of the average node degree, $\bar{k}$:

+++

$$
E[\bar{k}] = \frac{1}{n} pn(n-1) = p(n-1).
$$

+++

There are two senses of "average" going on here: in each graph instance, you find the average degree, then you take the average (expectation, $E[\cdot]$) over all random instances. Here is the distribution of $\bar{k}$ over 10000 instances when its expected value is $4.0$:

```{code-cell} ipython3
:tags: []

n,p = 41,0.1
kbar = []
for iter in range(10000):
    ER = nx.erdos_renyi_graph(n,p,seed=iter+1)
    kbar.append(average_degree(ER))

sns.displot(x=kbar,bins=19);
```
<!-- 
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
 -->
