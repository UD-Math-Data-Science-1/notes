---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.10.3
kernelspec:
  display_name: 'Python 3.8.8 64-bit (''base'': conda)'
  language: python
  name: python3
---
# Decision trees

A decision tree is much like playing "Twenty Questions." A question is asked, and the answer reduces the possible results, leading to a new question. **CART** (Classification And Regression Tree) is a popular method for systematizing the idea.

Given samples $\bfx_1,\ldots,\bfx_n$ and labels $y_1,\ldots,y_n$, the immediate goal is to partition the samples into subsets whose labels are as uniform as possible. The process is then repeated recursively on the subsets. Defining a measurement of label uniformity is a key step. 

Let $S$ be a subset of the samples, given as a list of indices into the original set. Suppose there are $K$ unique labels, which we denote $1,2,\ldots,K$. Define

$$
p_k = \frac{1}{ |S| } \sum_{i\in S} \mathbb{1}_k(y_i),
$$

where $|S|$ is the number of elements in $S$ and $\mathbb{1}_k$ is the **indicator function**

$$
\mathbb{1}_k(t) = \begin{cases} 
  1, & \text{if } t=k, \\ 
  0, & \text{otherwise.}
$$

In words, $p_k$ is the proportion of elements in $S$ that have label $k$. Then the **Gini impurity** is defined as 

$$
H(S) = \sum_{k=1}^K p_k(1-p_k).
$$

If one of the $p_k$ is 1, then the others are all zero and $H(S)=0$. This is considered optimal. At the other extreme, if $p_k=1/K$ for all $k$, then $H(S)=(K-1)/K$, which is the maximum value.

Now we can describe the partition process. If $j$ is a dimension (feature) number and $\theta$ is a numerical threshold, then the sample set can be partitioned into complementary sets $S_L$, in which $x_j \le \theta$, and $S_R$, in which $x_j > \theta$. Define the quality measure

$$
Q(j,\theta) = |S| H(S) + |T| H(T).
$$

Choose the $(j,\theta)$ that minimize $Q$, and then recursively partition $S$ and $T$.

## Toy example

We create a toy dataset with 20 random points, with two subsets of 10 that are shifted left/right a bit.

```{code-cell}
---
tags: [hide-input]
---
import numpy as np
from numpy.random import default_rng

rng = default_rng(1)
x1 = rng.random((10,2))
x1[:,0] -= 0.25
x2 = rng.random((10,2))
x2[:,0] += 0.25
X = np.vstack((x1,x2))
y = np.hstack(([1]*10,[2]*10))

import seaborn as sns
df = pd.DataFrame({"x1":X[:,0],"x2":X[:,1],"y":y})
sns.scatterplot(data=df,x="x1",y="x2",hue="y")
```

Now we create a decision tree for these samples.

```{code-cell}
from sklearn import tree
t = tree.DecisionTreeClassifier(max_depth=3)
t.fit(X,y)

tree.plot_tree(t,feature_names=["x1","x2"]);
```

The root of the tree (at the top) shows that the best split was found at the vertical line $x_1=0.644$. To the right of that line is a Gini value of zero: 8 samples, all with label 2. Thus, any future prediction by this tree will immediately return label 2 if the first feature of the input exceeds 0.644. Otherwise, it moves to the left child node and tests whether the second feature is greater than $0.96$. This splits along a horizontal line, above which there is a single sample with label 2. And so on.

Notice that the bottom right node has a nonzero Gini impurity. This node could be partitioned, but the classifier was constrained to stop at a depth of 3. If a prediction ends up here, then the classifier returns label 1, which is the most likely outcome.

Because we can follow the decision tree's logic step by step, we say it is highly **interpretable**. The transparency of the prediction algorithm is an attractive aspect of decision trees. 

## Penguin data

We return to the penguins. There is no need to standardize the columns for a decision tree, because each dimension is considered on it own.

```{code-cell}
import pandas as pd
pen = sns.load_dataset("penguins")
pen = pen.dropna()
X = pen[["bill_length_mm","bill_depth_mm","flipper_length_mm","body_mass_g"]]
y = pen["species"]
```

We again get some interesting information from a tree with limited depth.

```{code-cell}
t = tree.DecisionTreeClassifier(max_depth=2)
t.fit(X,y)

tree.plot_tree(t);
```

The most determinative feature for identifying the species is the flipper length. If it exceeds 206.5 mm, then the penguin is rather likely to be a Gentoo, and a further measurement of the bill depth settles that matter. Etc. Even for this shallow tree, two of the nodes at the bottom have small Gini impurity.

Here is a more systematic study of the accuracy performance.

```{code-cell}
from sklearn.model_selection import cross_val_score,KFold

D = range(1,8)
score_mean,score_std = [],[]
kf = KFold(n_splits=10,shuffle=True,random_state=1)
for d in D:
    t = tree.DecisionTreeClassifier(max_depth=d)
    scores = cross_val_score(t,X,y,cv=kf)
    score_mean.append(scores.mean())
    score_std.append(scores.std())

pd.DataFrame({"depth":D,"accuracy mean":score_mean,"accuracy std":score_std})
```

## Limitations

Decision trees depend sensitively on the sample locations. A small change can completely rewrite large parts of the tree, which gives a caveat about interpretation. They are also biased classifiers if the labels within the data set are not well-balanced between the classes Perhaps the greatest limitation to CART is that the partition algorithm, which is *greedy* by doing the best thing at the moment, does not necessarily find a globally optimal tree, or even a nearby one. 

To deal with these issues, one can use a **random forest**, which consists of many trees trained on subsets of the original features and datasets.   
