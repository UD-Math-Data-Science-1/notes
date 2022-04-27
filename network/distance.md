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

```{code-cell} ipython3
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
```

# Distance

The *small-world phenomenon* is, broadly speaking, the observation that any two people in a group can be connected by a surprisingly short path of acquaintances. This concept appears, for instance, in the "Bacon number" game, where actors are nodes, appearing in the same movie creates an edge between them, and one tries to find the distance between Kevin Bacon and some other designated actor. 

The **distance** between two nodes in a connected graph is the number of edges in the shortest path between them. For example, in a complete graph, the distance between any pair of distinct nodes is 1, since all possible pairs are connected by an edge.

```{code-cell} ipython3
K5 = nx.complete_graph(5)
dist = [ nx.shortest_path_length(K5,0,i) for i in K5.nodes ]
print("Distance from node 0:",dist)
```

The maximum distance over all pairs of nodes in a graph is called its **diameter**. Since this value depends on an extreme outlier in the distribution of distances, we often preferr to use the **average distance** as a measure of how difficult it is to connect two randomly selected nodes.

For example, here is a wheel graph:

```{code-cell} ipython3
W = nx.wheel_graph(7)
nx.draw(W,with_labels=True,node_color="lightblue")
```

No node is more than two hops away from another (if the first hop is to node 0), so the diameter of this graph is 2. The average distance is somewhat smaller. This graph is so small that we can easily find the entire matrix of pairwise distances. The matrix is symmetric, so it's only necessary to compute its upper triangle.

```{code-cell} ipython3
nodes = list(W.nodes)
n = len(nodes)
D = np.zeros((n,n),dtype=int)
for i in range(n):
    for j in range(i+1,n):
        D[i,j] = nx.shortest_path_length(W,nodes[i],nodes[j]) 

print(D)
```

To get the average distance, we can sum over all the entries and divide by $\binom{n}{2}$:

```{code-cell} ipython3
print("average distance:",2*D.sum()/(n*(n-1)))
```

There is a convenience function for computing this average. (It becomes slow as $n$ grows, though.)

```{code-cell} ipython3
print("average distance:",nx.average_shortest_path_length(W))
```

## ER graphs

If we want to compute distances within ER random graphs, we quickly run into a problem: an ER graph may not have a path between every pair of nodes:

```{code-cell} ipython3
n,p = 101,1/25
ER = nx.erdos_renyi_graph(n,p,seed=0)
nx.draw(ER,node_size=50)
```

We say that such a graph is not **connected**. When no path exists between two nodes, the distance between them is either undefined or infinite. NetworkX will give an error if we try to compute the average distance in a disconnected graph:

```{code-cell} ipython3
nx.average_shortest_path_length(ER)
```

One way to cope with this eventuality is to decompose the graph into **connected components**, a disjoint separation of the nodes into connected subgraphs. We can use `nx.connected_components` to get node sets for each component.

```{code-cell} ipython3
[len(cc) for cc in nx.connected_components(ER)]
```

The result above tells us that removing the lone unconnected node in the ER graph leaves us with a connected component. We can always get the largest component with the following idiom:

```{code-cell} ipython3
ER_sub = ER.subgraph( max(nx.connected_components(ER), key=len) )
print(ER_sub.number_of_nodes(),"nodes in largest component")
```

Now the average path length is a valid computation.

```{code-cell} ipython3
nx.average_shortest_path_length(ER_sub)
```

Let's use this method to examine average distances within ER graphs of a fixed type.

```{code-cell} ipython3
n,p = 121,1/20
dbar = []
for iter in range(100):
    ER = nx.erdos_renyi_graph(n,p,seed=iter+5000)
    ER_sub = ER.subgraph( max(nx.connected_components(ER), key=len) )
    dbar.append(nx.average_shortest_path_length(ER_sub))

print("average distance in the big component of ER graphs:")
sns.displot(x=dbar,bins=13);
```

The chances are good, therefore that any message could be passed along in three hops or fewer (within the big component). In fact, theory states that as $n\to\infty$, the mean distance in ER graphs is expected to be approximately 

```{math}
:label: eq-small-world-ERdistance
\frac{\ln(n)}{\ln(\bar{k})}.
```

For $n=121$ and $\bar{k}=6$ as in the experiment above, this value is about 2.68.

## Twitch network

Let's consider distances within the Twitch network.

```{code-cell} ipython3
twitch = nx.read_edgelist("musae_edges.csv",delimiter=',',nodetype=int)
n,e = twitch.number_of_nodes(),twitch.number_of_edges()
kbar = 2*e/n
print(n,"nodes and",e,"edges")
print(f"average degree is {kbar:.3f}")
```

Computing the distances between all pairs of nodes in this graph would take a rather long time, so we will sample some pairs randomly.

```{code-cell} ipython3
from numpy.random import default_rng
rng = default_rng(1)

# Compute the distance between a random pair of distinct nodes:
def pairdist(G):
    n = nx.number_of_nodes(G)
    i = j = rng.integers(0,n)
    while i==j: j=rng.integers(0,n)   # get distinct nodes
    return nx.shortest_path_length(G,source=i,target=j)

distances = [pairdist(twitch) for _ in range(50000)]
print("Pairwise distances in Twitch graph:")
sns.displot(x=distances,discrete=True)
print("estimated mean =",np.mean(distances))
```

Let's compare these results to ER graphs with the same size and average degree, i.e., with $p=\bar{k}/(n-1)$. The theoretical estimate from above gives

```{code-cell} ipython3
print("Comparable ER graphs expected mean distance:",np.log(n)/np.log(kbar))
```

The Twitch network might have a slightly stronger small-world effect than a random ER graph, but not dramatically so.

+++

## Watts–Strogatz graphs

The Watts–Strogatz model was originally proposed to explain small-world networks. We use the same $n$ as the Twitch network, and choose $k=10$ to get a similar average degree. 

```{code-cell} ipython3
results = []
seed = 44044
n,k = twitch.number_of_nodes(),10
for q in np.arange(0.1,0.76,0.05):
    for iter in range(10):
        WS = nx.watts_strogatz_graph(n,k,q,seed=seed)
        dbar = sum(pairdist(WS) for _ in range(60))/60
        results.append( (q,dbar) )
        seed += 7

results = pd.DataFrame(results,columns=["q","avg distance"])
print("Pairwise distances in WS graphs:")
sns.relplot(data=results,x="q",y="avg distance",kind="line");
```

We see from the above that pairwise distances decrease as the fraction of rewired edges grows, as we would expect. However, the values here are all well larger than the Twitch mean distance of 3.87, particularly at the value $q=0.42$ that we found in the previous section to match the average clustering. Thus, the Watts-Strogatz model may not be able to explain the Twitch network well. We will confirm that conclusion in the next section.
