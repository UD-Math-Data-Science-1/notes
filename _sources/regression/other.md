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
# Repurposed classifiers

Most classification algorithms have a regression counterpart. For two of the classification methods we looked at, these are easy to describe and understand.

## Nearest neighbors

To use kNN for regression, we can find the $k$ nearest examples as with classification, but replace voting on classes with averaging of values. As with the classification variant, we can choose the norm to use, and whether to weight the neighbors equally or by inverse distance. Also, it is again recommended to start with z-scores for the features rather than raw data; it is not necessary to standardize the target values being regressed.

As you can see in the following video, increasing $k$ flattens out the surface, decreasing variance while increasing bias.

```{raw} html
<video width=640 controls src="../_static/knn_regression.mp4"></video>
```

## Decision tree

Recall that a decision tree recursively divides the examples into subsets. Instead of using a measure of class impurity to determine the best split, the split is chosen to cause the greatest reduction in dispersion within the two subsets. The most obvious candidates for measuring dispersion are standard deviation and the mean absolute deviation (MAD). If $t_1,\ldots,t_m$ are elements of any set, then their MAD is given by

$$
\text{MAD} = \frac{1}{m} \sum_{i=1}^m | t_i - t_\text{med} |,
$$

where $t_\text{med}$ is the median value. As with classification, a proposal to split into subsets $S$ and $T$ is assigned the weighted score

$$
Q = |S| H(S) + |T| H(T),
$$

where $H$ is the chosen dispersion measure.

::::{prf:example}
:label: example-other-dtree
Choose the best split using the MAD criterion for the observations $x_i=i$, $i=1,\ldots,4$, where $y_1=2$, $y_2=-1$, $y_3=1$, $y_4=0$.

:::{dropdown} Solution
The original set has median $\frac{1}{2}$ and gets a weighted dispersion of $\frac{5}{2}(3+3+1+1)=20$. There are three ways to split the data, depending on where the partition falls in relation to the $x_i$.

* $S=\{2\},T=\{-1,1,0\}$: 

$$ Q &= 1\left[ \frac{1}{1}\left( |2-2| \right)  \right] +  3 \left[ \frac{1}{3}\left( | -1-0 | + |1-0| + |0-0|  \right)  \right]\\ &=  0 + 2 = 2.$$

* $S=\{2,-1\},T=\{1,0\}$: 

$$ Q &= 2\left[ \frac{1}{2}\left( \left| 2-\tfrac{1}{2} \right| + \left| -1-\tfrac{1}{2} \right| \right)  \right] +  2 \left[ \frac{1}{2}\left( \left|1-\tfrac{1}{2} \right| + \left|0-\tfrac{1}{2} \right|  \right)  \right]\\ 
&=  3 + 1 = 4.$$

* $S=\{2,-1,1\},T=\{0\}$: 

$$ Q &= 3\left[ \frac{1}{3}\left( \left| 2-1 \right| + \left| -1-1 \right|+ |1-1| \right)  \right] +  1 \left[ \frac{1}{1}\left(  \left|0-0 \right|  \right)  \right]\\ 
&=  3 + 0 = 3.$$

Thus, the first split above produces the smallest total dispersion.
:::
::::

To predict a value for a query $x$, we follow the tree until ending at a leaf, where we use the mean (if dispersion is STD) or median (if dispersion is MAD) of the examples in the leaf.

Here is a simple example based on two dimensions. 

```{code-cell}
---
tags: [hide-input]
---
import numpy as np
import pandas as pd
from numpy.random import default_rng

rng = default_rng(1)
x1 = rng.random((10,2))
x1[:,0] -= 0.25
x2 = rng.random((10,2))
x2[:,0] += 0.25
X = np.vstack((x1,x2))
y = np.exp(X[:,0]-2*X[:,1]**2+X[:,0]*X[:,1])

import seaborn as sns
df = pd.DataFrame({"x₁":X[:,0],"x₂":X[:,1],"y":y})
sns.scatterplot(data=df,x="x₁",y="x₂",hue="y");
```

Here is a shallow tree for regression on this data. The default choice is to use STD as the dispersion measure (called `squared_error` in sklearn).

```{code-cell}
from sklearn.tree import DecisionTreeRegressor,plot_tree
dt = DecisionTreeRegressor(max_depth=2)
dt.fit(X,y)

from matplotlib.pyplot import figure
figure(figsize=(18,11),dpi=160)
plot_tree(dt,feature_names=["x₁","x₂"])
```

All of the original samples end up in one of the four leaves. We can find out which goes where using `apply`:

```{code-cell}
leaf = dt.apply(X)
print(leaf)
```

With some pandas grouping, we can find out the mean value for the samples within each leaf:

```{code-cell}
leaves = pd.DataFrame(zip(y,leaf),columns=["y","leaf"])
leaves.groupby("leaf")["y"].mean()
```

All values of the regressor will be one of the four values above:

```{code-cell}
print(dt.predict(X))
```

## Case study

Here again is the diabetes data set, and the performance of a multilinear regressor on it.

```{code-cell}
from sklearn import datasets
diabetes = datasets.load_diabetes(as_frame=True)["frame"]
X = diabetes.drop("target",axis=1)
y = diabetes["target"]

from sklearn.model_selection import train_test_split
X_tr,X_te,y_tr,y_te = train_test_split(X,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_tr,y_tr)
print("linear model CofD score:",lm.score(X_te,y_te))
```

We will do a cursory grid search over some useful hyperparameters in order to get the best regressors from a decision tree and k-nearest neighbors. The data columns in this frame are already standardized, so there is no need to set up a pipeline for preprocessing. Note that the DT and kNN regressors, like the linear regressor, uses coefficient of determination in their `score` methods. 

```{code-cell}
from sklearn.model_selection import GridSearchCV,KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

kf = KFold(n_splits=6,shuffle=True,random_state=3)
grid = { "max_depth":range(2,12), "criterion":["squared_error","absolute_error"] }
dt = GridSearchCV(DecisionTreeRegressor(random_state=5),grid)
dt.fit(X_tr,y_tr)

grid = { "n_neighbors":range(2,25), "weights":["uniform","distance"] }
knn = GridSearchCV(KNeighborsRegressor(),grid)
knn.fit(X_tr,y_tr)

print("Best decision tree CofD:",dt.score(X_te,y_te))
print("Best kNN CofD:",knn.score(X_te,y_te))
```

As you can see, kNN is no better than the linear model for this dataset. Moreover, the coefficients of the linear model give some potentially useful information, and the linear model can be improved somewhat with regularization, while kNN offers no significant advantages here.