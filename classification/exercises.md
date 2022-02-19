# Exercises

For these exercises, you may use computer help to work on a problem, but your answer should be self-contained without reference to computer output (unless stated otherwise).

1. Here is a confusion matrix for a classifier of meme dankness. 

    ```{image} dankness.png
    :alt: confusion matrix
    :width: 360px
    :align: center
    ```

    Calculate the **(a)** recall, **(b)** precision, **(c)** specificity, **(d)** accuracy, and **(e)** *F*₁ score of the classifier.

2. Here is a confusion matrix for a classifier of ice cream flavors. 

    ```{image} flavors.png
    :alt: confusion matrix
    :width: 500px
    :align: center
    ```
    
    **(a)** Calculate the recall rate for chocolate.

    **(b)** Find the precision for vanilla. 

    **(c)** Find the accuracy for strawberry.
 
3. Find the Gini impurity of a set of six elements, of which 1 is labelled A, 2 are labelled B, and 3 are labelled C.

4. Given $x_i=i$ for $i=0,\ldots,5$, with labels (from left to right) A,B,B,B,A,A, find an optimal partition threshold. 

5. Carefully sketch the set of all points in $\real^2$ whose 1-norm distance from the origin equals 1. This is a *Manhattan unit circle*.

6. Three points in the plane lie at the vertices of an equilateral triangle. Carefully sketch the decision boundaries of $k$-nearest neighbors with $k=1$, using the 2-norm. 

7. Define points on an ellipse by $x_k=a\cos(\theta_k)$ and $y_k=b\sin(\theta_k)$, where $a$ and $b$ are positive and $\theta_k=2k\pi/8$ for $k=0,1,\ldots,7$. Show that if the $x_k$ and $y_k$ are standardized into z-scores, then the resulting points all lie on a circle centered at the origin. (Standardizing points into z-scores is sometimes called *sphereing* them.)

8.  Using the formulas from section 3.5, find the distance from the point $(-1,2)$ to the line $y=3x-4$. (First find $w_1$, $w_2$, and $b$ for the line.)

9. If $\bfu = [1,2,3]$ and $\bfv=[4,5,-6]$, compute $\bfu^T \bfv$. 

10. If $\bfv$ is a nonzero vector, then the **projection** of any vector $\bfu$ onto $\bfv$ is 
    
    $$
    \text{proj}_\bfv(\bfu) = \frac{\bfu^T\bfv}{\bfv^T\bfv}\, \bfv.
    $$

    Show that the inner product between $\bfu$ and $\bfu - \text{proj}_\bfv(\bfu)$ is zero.

11. Find an equation for the plane normal to the vector $[-1,2,3]$ and passing through the point $(1,0,2)$.
