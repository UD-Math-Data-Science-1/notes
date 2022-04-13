# Exercises

1. Using only the three axioms of a distance metric, prove that $\dist(\bfx,\bfy) \ge 0$ for all vectors $\bfx$ and $\bfy$. (Hint: apply the triangle inequality to go from $\bfx$ to $\bfy$ and back again.)
2. Prove that the angular distance between any nonzero vector and itself is zero.
3. Find a counterexample showing that cosine distance does not satisfy the triangle inequality. (Hint: it's enough to consider some simple vectors in two dimensions.)
4. Let $c$ be a positive number, and consider the 12 sample points $\{(\pm c,\pm j): j=1,2,3\}$. One way to cluster the sample points, which we designate as clustering $\alpha$, is to split according to the sign of $x_1$. Another way, which we designate as clustering $\beta$, is to split according to the sign of $x_2$. Compute the inertia of both clusterings. For which values of $c$, if any, does clustering $\alpha$ have less inertia than clustering $\beta$?
5. Here is a distance matrix for points $\bfx_1,\ldots,\bfx_5$. 
    
    $$
    \left[
    \begin{array}{ccccc}
    0 & 2 & 4 & 5 & 6 \\
    2 & 0 & 2 & 3 & 4 \\
    4 & 2 & 0 & 1 & 2 \\
    5 & 3 & 1 & 0 & 1 \\
    6 & 4 & 2 & 1 & 0 \\
    \end{array}
    \right]
    $$

    Compute the average linkage between the clusters with index sets $C_1=\{1,3\}$ and $C_2=\{2,4,5\}$. 

6. Perform by hand an agglomerative clustering for the values $2,4,5,8,12$ using single linkage. This means finding the four merge steps needed to convert five singleton clusters into one global cluster.
7. Here are some sample points in the plane.

    ```{image} ../_static/plane_points.svg
    :alt: plane points
    :width: 500px
    :align: center
    ```

    (a) Find the $\epsilon$-neighborhoods of every point if $\epsilon=1.8$. 

    (b) Using $N_\text{min}=4$, which points are DBSCAN core points?

    (c) Starting with the lowest unassigned core point for each cluster, find the DBSCAN clustering with the above hyperparameters. Sketch the result, showing the clusters and noise points.