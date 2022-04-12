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

# Random graphs

```{code-cell} ipython3
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
```

One way of trying to understand a new network is by comparing it to networks constructed by random processes governed by fairly simple rules. if the network has quantitative properties similar to those of some random family, we may suppose that the construction rules of that family have some bearing on the way our network came into being.

+++

An **Erdős-Rényi graph** includes each individual possible edge with a fixed probability $p$. When one refers to a "random graph" without any additional context, this is usually the type that is meant.

```{code-cell} ipython3
n,p = 60,0.08
ER = nx.erdos_renyi_graph(n,p,seed=1)
print(er.number_of_nodes(),"nodes,",ER.number_of_edges(),"edges")
nx.draw_circular(ER,node_size=40)
```

Since there are $\binom{n}{2}$ possible edges in an undirected graph on $n$ nodes, the mean number of edges in an ER graph is 

$$
\frac{pn(n-1)}{2}.
$$

This fact is usually stated in different terms. The **degree** of a node is the number of edges that have the node as an endpoint. The `degree` property of a graph returns an iterator that can be collected into a list or frame.

```{code-cell} ipython3
degrees = pd.DataFrame(ER.degree,columns=["node","degree"])
degrees.head()
```

In any graph, if we add the degrees of all the nodes, then we must get twice the number of edges. Thus, in an ER graph, the average node degree is, on average,

$$
\bar{k} = \frac{1}{n} pn(n-1) = p(n-1).
$$

```{code-cell} ipython3
:tags: []

print("Average degree in ER graph above:",degrees["degree"].mean())
print("Theoretical expectation:",p*(n-1))
```

Note that we are talking here about the average (over all ER graphs of type $(n,p)$) of the value of the average (over the nodes in one graph) degree. Here is the distribution of $\bar{k}$ over many instances.

```{code-cell} ipython3
:tags: []

n,p = 41,0.1
kbar = []
for i in range(5000):
    ER = nx.erdos_renyi_graph(n,p)
    deg = pd.DataFrame(ER.degree,columns=["node","degree"])["degree"]
    kbar.append(deg.mean())

sns.displot(x=kbar,bins=19);
```

## Degree distribution

As we know, means of distributions do not always tell the entire story. For example, here is the distribution of the degrees of all the nodes in our Twitch network example.

```{code-cell} ipython3
:tags: []

twitch = nx.read_edgelist("musae_edges.csv",delimiter=',',nodetype=int)
twitch_degrees = pd.DataFrame(twitch.degree,columns=["node","degree"])
hist = sns.displot(data=twitch_degrees,x="degree")
```

We can find the average degree of this network and create an ER graph with the same expected average degree.

```{code-cell} ipython3
n = twitch.number_of_nodes()
kbar = degrees["degree"].mean()
p = kbar/(n-1)
ER = nx.erdos_renyi_graph(n,p)
degrees = pd.DataFrame(ER.degree,columns=["node","degree"])
hist = sns.displot(data=degrees,x="degree")
```

+++ {"tags": []}

It's obvious now that the ER graph is nothing like the Twitch graph. While nobody in the ER graph has more than 30 friends, a few nodes in the Twitch network have hundreds of friends.

+++

The behavior of the Twitch degree distribution gets very interesting when the axes are transformed to use log scales:

```{code-cell} ipython3
hist = sns.displot(data=twitch_degrees,x="degree",log_scale=True)
hist.axes[0,0].set_yscale("log")
```

For degrees between 10 and several hundred, the counts lie nearly on a straight line. That is, if $x$ is degree and $y$ is the node count at that degree, then

$$
\log(y) \approx  - a\cdot \log(x) + b,
$$

i.e.,

$$
y \approx B x^{-a},
$$

for some $a > 0$. This relationship is known as a **power law**. Many social networks follow to some extent a power-law distrubution of node degrees. (The extent to which this statement is true is a hot debate within network science.)

+++

We can get a fair estimate of the constants $B$ and $a$ in the power law by doing a least-squares fit on the logs of $x$ and $y$. First, we need the counts:

```{code-cell} ipython3
y = twitch_degrees["degree"].value_counts()
counts = pd.DataFrame({"degree":y.index,"count":y.values})
counts = counts[(counts["count"] > 10) & (counts["count"] < 200)];
```

Now we will get additional columns by log transformations. (Note: the `np.log` function is the natural logarithm.)

```{code-cell} ipython3
counts[["log_x","log_y"]] = counts.transform(np.log)
```

Now we use `sklearn` for a linear regression.

```{code-cell} ipython3
:tags: []

from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(counts[["log_x"]],counts["log_y"])
print("slope",lm.coef_[0],"and intercept",lm.intercept_)
```

If the average degree in any graph is less than 1, then at least one node must have degree less than one. Therefore, the graph is not connected. For ER graphs, $\bar{k} = 1$ marks a crucial transition. Let's fix $n$ and look at the size of the largest connected component of ER graphs as $p$ varies.

```{code-cell} ipython3
n = 201
results = pd.DataFrame({"p":[],"kbar":[],"size":[]})
for p in np.arange(0.0001,0.025,0.0004):
    for i in range(50):
        ER = nx.erdos_renyi_graph(n,p)
        largest_cc = max(len(g) for g in nx.connected_components(ER))
        results = pd.concat((results,
                             pd.DataFrame({"p":[p],"kbar":p*(n-1),"size":largest_cc})),
                            ignore_index=True)
```

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: true
tags: []
---
sns.relplot(data=results,x="kbar",y="size",kind="line");
```

The plot above suggests that as $\bar{k}$ increases past 1, there is often a "giant component" that dominates the graph.

```{code-cell} ipython3
:tags: []

degrees = pd.DataFrame(twitch.degree,columns=["node","degree"])
degrees
```

```{code-cell} ipython3
nx.draw(nx.cycle_graph(5))
```

A **Watts–Strogatz graph** tries to model the "small-world" phenomenon of social networks, where most members can be connected by a suprirsingly short walk along edges. A WS graph has three parameters: $n$, an even integer $k$, and a probability $p$. 

Imagine $n$ nodes arranged in a circle. Connect each node with an edge to each of its $k/2$ left neighbors and $k/2$ right neighbors. Now visit each node $i$ in turn. For each edge from $i$ to a neighbor, with probability $p$ replace it with an edge between $i$ and a node chosen at random from all the nodes $i$ is not currently connected to.

+++

The first value, which is both the slope of the line and the exponent of $x$ the power law, is the most interesting part. It estimates that the degree counts vary as $Bx^{-2.1}$ over a wide range of degrees.

+++

A certain type of random graph known as a **Barabási–Albert** graph is constructed by connecting one node at a time. Edges are added randomly, but preference is given to connect to nodes that already have higher degree (i.e., are more "popular"). Because of this rule, there is a natural tendency to develop a few hubs of high degree.

```{code-cell} ipython3
bag = nx.barabasi_albert_graph(50,2,seed=0)
bag_deg = pd.DataFrame(bag.degree,columns=["node","degree"])

nx.draw(bag,node_size=8*bag_deg["degree"],node_color="red")
```

When we scale the construction up to the size and average degree of the Twitch network, a power-law distribution emerges.

```{code-cell} ipython3
bag = nx.barabasi_albert_graph(n,round(p*n),seed=0)
bag_deg = pd.DataFrame(bag.degree,columns=["node","degree"])
hist = sns.displot(data=bag_deg,x="degree",log_scale=True)
hist.axes[0,0].set_yscale("log")
```

Compared to selecting from all possible edges with equal probabiliy, the Barabási–Albert model is considered a more plausible idealization of how social networks form.

+++

## Distance

+++

The **distance** between two nodes in a connected graph is the number of edges in the shortest path between them. This concept appears in the "Bacon number" game, where actors are nodes, appearing in the same movie creates an edge between them, and one tries to find the distance between Kevin Bacon and some other designated actor. 

The number of pairs of nodes in our Twitch network is moderately large.

```{code-cell} ipython3
:tags: []

print(n,"nodes create",(n-1)*n//2,"possible pairings")
```

It would take a while on a typical laptop to compute the distances between all such pairs. Instead, we will just randomly sample a healthy number of them.

```{code-cell} ipython3
from numpy.random import default_rng
rng = default_rng(1)
def pairdist(G,n):
    i = j = rng.integers(0,n)
    while i==j: j = rng.integers(0,n)
    return nx.shortest_path_length(G,source=i,target=j)

dist = [pairdist(twitch,n) for _ in range(100000)]
distances = pd.Series(dist)
```

```{code-cell} ipython3
distances.value_counts()
```

With over 7100 nodes, and an average of about 10 edges per node, the average distance between nodes is less than 4:

```{code-cell} ipython3
distances.mean()
```

Node distances are related to the notion of "six degrees of separation" between any two people on Earth. This is a statement about the **diameter**, or maximum distance, of the human friendship network. In network analysis, this notion is called the *small-word phenomenon*. It's usually stated in terms of the mean degree, not the maximum, since the diameter can be determined by a single pair. (Even the little Twitch network above has a diameter of at least 9.) For a random network on $n$ nodes that have average degree $d$, the mean distance is approximately $\ln(n)/d$. For a scale-free network, it is even less.

```{code-cell} ipython3
from numpy.random import default_rng
rng = default_rng(1)

distances = pd.Series([pairdist(bag,n) for _ in range(100000)])
```

```{code-cell} ipython3
distances.mean()
```

## Clustering

+++

The small-world question is related to another type of network measurement: **clustering**. There are many ways to assess clustering, but we will use the **local clustering coefficient**. 

Suppose that node $i$ is adacent to $k$ other nodes, called its **neighborhood** (or level-1 ego graph). The nodes in the neighborhood have $k(k-1)/2$ possible edges between them. Suppose that there are $m$ edges between members of the neighborhood (exclusive of node $i$). The clustering coefficient of node $i$ is defined as

$$
c_i = \frac{2m}{k(k-1)}.
$$

If $c_i=0$, the neighborhood is starlike: none of the neighbors "talk to" (i.e., are adjacent to) each other. At the other extreme, if $c_i=1$, then the subgraph of the neighborhood is a complete graph. 

For example, here is the neighborhood of node 400.

```{code-cell} ipython3
nbhood = nx.ego_graph(twitch,400)
nx.draw(nbhood,with_labels=True,node_size=800,node_color="yellow")
```

The plot reveals that the neighborhood has only 3 edges aside from those linking to node 400. Since there are 9 nodes in the neighborhood, this gives a clustering of 

$$
\frac{6}{9\cdot 8} = \frac{1}{12}.
$$

There is a function for computing this value, of course.

```{code-cell} ipython3
nx.clustering(twitch,400)
```

The mean clustering coefficient is one way to assess the small-worldness of a graph.

```{code-cell} ipython3
pd.Series(nx.clustering(twitch)).mean()
```

How do we put this number in context? The random BA graph has a much lower value.

```{code-cell} ipython3
pd.Series(nx.clustering(bag)).mean()
```

This confirms that the BA construction does nothing to build community structure, only hubs. A Watts–Strogatz graph, on the other hand, starts from communities and just rewires some of the edges to create shortcuts. If we start from neighborhoods of size 10, the mean clustering is high:

```{code-cell} ipython3
ws = nx.watts_strogatz_graph(n,10,0,seed=2)
pd.Series(nx.clustering(ws)).mean()
```

However, the mean path length in that graph would be large, since there are no far-flung direct connections. But if we allow 40% of the links to be converted to random destinations, then the clustering is similar to the Twitch graph:

```{code-cell} ipython3
ws = nx.watts_strogatz_graph(n,10,0.4,seed=11)
pd.Series(nx.clustering(ws)).mean()
```

The mean distance is also comparable:

```{code-cell} ipython3
rng = default_rng(1)
pd.Series([pairdist(ws,n) for _ in range(50000)]).mean()
```

As we computed above, the Twitch network has an even smaller mean distance, making it a bit more efficient at connecting nodes by paths.
