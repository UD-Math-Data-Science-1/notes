# Classification

Machine learning is the use of data to tune algorithms for making decisions or predictions. Unlike deduction based on reasoning from principles governing the application, machine learning is a "black box" that just adapts via training.

We divide machine learning into three major forms: 

* **Supervised learning** The training data only examples that include the answer (or **label**) we expect to get. The goals are to find important effects and/or to predict labels for previously unseen examples.
* **Unsupervised learning** The data is unlabeled, and the goal is to discover structure and relationships inherent to the data set.
* **Reinforcement learning** The data is unlabeled, but there are known rules and goals that can be encouraged through penalties and rewards.

We start with supervised learning, which can be subdivided into two major areas:

* **Classification**, in which the algorithm is expected to choose from among a finite set of options.
* **Regression**, in which the algorithm should predict the value of a quantitative variable.

Most algorithms for one of these problems have counterparts in the other. 

<!-- Regression methods can typically be used as classifiers, by thresholding or binning the result. For example, in a yes/no situation, a predictor of probability can be used to decide "yes" if its probability exceeds 50%. -->

Each observation in the training set is supplied by a **feature vector** of length $d$. (Recall that a vector is a series indexed by integers. Mathematically the index starts at 1, but in Python it starts at zero.) The components of a feature vector are assumed to be real numbers. A set of $n$ observations can be represented as a **feature matrix** $bfX$ with $n$ rows and $d$ columns. Using subscripts to represent the index values, we can write

$$
\bfX = \begin{bmatrix}
X_{11} & X_{12} & \cdots & X_{1d} \\
X_{21} & X_{22} & \cdots & X_{2d} \\
\vdots & \vdots && \vdots \\ 
X_{n1} & X_{n2} & \cdots & X_{nd} 
\end{bmatrix}.
$$

Note that *the first index refers to the row number, and the second to the column number*. The rows may be referred to as instances, examples, samples, etc.

We can also collect the associated training labels as the **label vector**

$$
\bfy = \begin{bmatrix} 
y_1 \\ y_2 \\ \vdots \\ y_n
\end{bmatrix}
$$

```{note}
In linear algebra, the default shape for a vector is as a single column. In Python, a vector doesn't exactly have a row or column orientation, though when it matters, a row shape is usually preferred.
```

```{prf:example}
Suppose we want to train an algorithm to predict whether a basketball shot will score. For one shot, we might collect three coordinates to represent the launch point, three to represent the launch velocity, and three to represent the initial angular rotation (axis and magnitude). Thus each shot will require a feature vector of length 9.
```
