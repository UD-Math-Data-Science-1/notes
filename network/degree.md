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

# Degree distributions

```{code-cell} ipython3
import networkx as nx
import pandas as pd
import numpy as np
import seaborn as sns
```

As we know, means of distributions do not always tell the entire story. For example, here is the distribution of the degrees of all the nodes in our Twitch network.

```{code-cell} ipython3
:tags: []

twitch = nx.read_edgelist("musae_edges.csv",delimiter=',',nodetype=int)
twitch_degrees = pd.Series(dict(twitch.degree),index=twitch.nodes)
print("Twitch network degree distribution:")
sns.displot(data=twitch_degrees);
```

A few nodes in the network have hundreds of friends:

```{code-cell} ipython3
:tags: []

friend_counts = twitch_degrees.value_counts()  # histogram heights
friend_counts.sort_index(ascending=False)
```

These "gregarious nodes" or *hubs* create the heavy tail in the degree distribution.

+++

We can compare the above distribution to those in a collection of ER graphs with the same size and expected average degree.

```{code-cell} ipython3
n,e = twitch.number_of_nodes(),twitch.number_of_edges()
kbar = 2*e/n
p = kbar/(n-1)
degrees = []
for iter in range(3):
    ER = nx.erdos_renyi_graph(n,p,seed=111+iter)
    degrees.extend([ER.degree(i) for i in ER.nodes])

print("ER graphs degree distribution:")
sns.displot(data=degrees,discrete=True);
```

+++ {"tags": []}

Theory proves that the plot above converges to a *binomial distribution*. This is yet another indicator that the ER model does not explain the Twitch network well.

A WS graph likewise lacks the proper heavy tail in the Twitch degree distribution:

```{code-cell} ipython3
k,q = 10,0.42
degrees = []
for iter in range(3):
    WS = nx.watts_strogatz_graph(n,k,q,seed=222+iter)
    degrees.extend([WS.degree(i) for i in WS.nodes])

print("WS graphs degree distribution:")
sns.displot(data=degrees,discrete=True);
```

## Power-law distribution

The behavior of the Twitch degree distribution gets very interesting when the axes are transformed to use log scales:

```{code-cell} ipython3
hist = sns.displot(data=twitch_degrees,log_scale=True)
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

for some $a > 0$. This relationship is known as a **power law**. Many social networks seem to follow a power-law distribution of node degrees, to some extent. (The precise extent is a subject of hot debate.)

Note that the decay of $x^{-a}$ to zero as $x\to\infty$ is much slower than, say, the normal distribution's $e^{-x^2/2}$, or even just an exponential $e^{-cx}$. This last comparison is how a *heavy-tailed distribution* is usually defined. One effect is that there is a significant disparity between the mean and median values of the node degrees:

```{code-cell} ipython3
twitch_degrees.describe()
```

The summary above also shows that the standard deviation is much larger than the mean. This is another indication that the degree distribution is widely dispersed over orders of magnitude.

+++

We can get a fair estimate of the constants $B$ and $a$ in the power law by doing a least-squares fit on the logs of $x$ and $y$. First, we need the counts:

```{code-cell} ipython3
y = twitch_degrees.value_counts()
counts = pd.DataFrame({"degree":y.index,"count":y.values})
counts = counts[(counts["count"] > 10) & (counts["count"] < 200)];
counts.head(6)
```

Now we will get additional columns by log transformations. (Note: the `np.log` function is the natural logarithm.)

```{code-cell} ipython3
counts[["log_degree","log_count"]] = counts.transform(np.log)
```

Now we use `sklearn` for a linear regression.

```{code-cell} ipython3
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(counts[["log_degree"]],counts["log_count"])
lm.coef_[0],lm.intercept_
```

The first value, which is both the slope of the line and the exponent of $x$ in the power law, is the most interesting part. It estimates that the degree counts vary as $Bx^{-2.1}$ over a wide range of degrees.

+++

## Barabási–Albert graphs

A random **Barabási–Albert** graph (BA graph) is constructed by starting with a small seed network and connecting one node at a time with $m$ new edges to it. Edges are added randomly, but higher probability is given to connect to nodes that already have higher degree (i.e., are more "popular"), a concept known as *preferential attachment*. Because of this rule, there is a natural tendency to develop a few hubs of high degree.

```{code-cell} ipython3
BA = nx.barabasi_albert_graph(100,2,seed=0)
BA_degrees = pd.Series(dict(BA.degree),index=BA.nodes)
nx.draw(BA,node_size=8*BA_degrees,node_color="red")
```

When we match these graphs to the size and average degree of the Twitch network, a power-law distribution emerges.
Since we add $m$ edges (almost) $n$ times, the expected average degree is $2mn/n=2m$. Therefore, in the BA construction we want to choose 

$$
m \approx \frac{\bar{k}}{2}. 
$$


```{code-cell} ipython3
:tags: []

m = round(kbar/2)
BA = nx.barabasi_albert_graph(n,m,seed=5)
BA_degrees = pd.Series(dict(BA.degree),index=BA.nodes)
hist = sns.displot(data=BA_degrees,log_scale=True)
hist.axes[0,0].set_yscale("log")
```

Theory predicts that the exponent of the power-law distribution in a BA graph is $-3$.

```{code-cell} ipython3
y = BA_degrees.value_counts()
counts = pd.DataFrame({"degree":y.index,"count":y.values})
counts = counts[(counts["count"] > 10) & (counts["count"] < 100)];
counts[["log_degree","log_count"]] = counts.transform(np.log)
lm = LinearRegression()
lm.fit(counts[["log_degree"]],counts["log_count"])
print("exponent of power law:",lm.coef_[0])
```

Let's check distances and clustering, too. As a reminder, the mean distance in the Twitch network is approximately:

```{code-cell} ipython3
from numpy.random import default_rng
rng = default_rng(1)

def pairdist(G):
    n = nx.number_of_nodes(G)
    i = j = rng.integers(0,n)
    while i==j: j=rng.integers(0,n)   # get distinct nodes
    return nx.shortest_path_length(G,source=i,target=j)

print("Mean distance in Twitch graph:",sum(pairdist(twitch) for _ in range(4000))/4000)
```

Now we repeat that for some BA graphs.

```{code-cell} ipython3
dbar = []
seed = 911
for iter in range(10):
    BA = nx.barabasi_albert_graph(n,m,seed=seed)
    d = sum(pairdist(BA) for _ in range(200))/200
    dbar.append(d)
    seed += 1

print("Mean distance in BA graphs:",np.mean(dbar))
```

Not bad! Now, let's check the clustering. For Twitch, we have:

```{code-cell} ipython3
print("Mean clustering in Twitch graph:",nx.average_clustering(twitch))
```

And for BA, we get

```{code-cell} ipython3
cbar = []
seed = 59
for iter in range(20):
    BA = nx.barabasi_albert_graph(n,m,seed=seed)
    cbar.append(nx.average_clustering(BA))
    seed += 1
    
print("Mean clustering in BA graphs:",np.mean(cbar))
```

The BA model is our closest approach so far, but it fails to produce the close-knit neighbor subgraphs that we find in the Twitch network and the WS model.
