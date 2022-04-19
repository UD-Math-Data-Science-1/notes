---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---
# Exercises

1. For each graph, give the number of nodes, the number of edges, and the average degree.

    **(a)** The complete graph $K_6$.

    **(b)** 
    ```{image} nws.svg
    :alt: NWS graph
    :width: 300px
    :align: center
    ```

    **(c)** 
    ```{image} ladder.svg
    :alt: ladder graph
    :width: 300px
    :align: center
    ```

2. Give the adjacency matrix for the graph in Exercise 1(c).

3. For the graph below, draw the ego graph of **(a)** node 4 and **(b)** node 8.

    ```{image} ba.svg
    :alt: BA graph
    :width: 300px
    :align: center
    ```

4. To construct an Erdős-Rényi graph on 25 nodes with expected average degree 8, what should the edge inclusion probability $p$ be?

5. Find the diameter of the graphs in Exercise 1.

6. Find the clustering coefficient for each node in the following graph:

    ```{image} lolly.svg
    :alt: Lollipop graph
    :width: 300px
    :align: center
    ```

7. The Watts–Strogatz construction starts with a *ring lattice* in which the nodes are arranged in a circle and each is connected to its $k$ nearest neighbors ($k/2$ on each side). Find the clustering coefficient of an arbitrary node in the ring lattice.