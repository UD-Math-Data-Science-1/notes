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
# Nearest neighbors

Our first learning algorithm is conceptually very simple: Given a new point to classify, survey the nearest examples and choose the most frequent class. This is called the **$k$ nearest neighbors** (KNN) algorithm, where $k$ is the number of neighboring examples to survey.

## Norms

The existence of "closest" examples means that we need to define a notion of distance in spaces of any dimension. Let $\real^d$ be the space of vectors with $d$ real components, and let $\bfzero$ be the vector of all zeros.

```{prf:definition}
A **norm** is a function $\|\bfx\|$ on $\real^d$ that satisfies the following properties:

$$
\|\bfzero\| &= 0,  \\ 
\|\bfx\| &> 0 \text{ if $\bfx$ is a nonzero vector}, \\ 
\|c\bfx\| &= |c| \, \|x\| \text{ for any real number $c$}, \\ 
\|\bfx + \bfy \| &\le \bfx + \bfy.
$$
```

The last inequality above is called the **triangle inequality**. It turns out that these four characteristics are all we expect from a function that behaves like a distance. 

We will encounter two different norms:

* The 2-norm, $\|\bfx\|_2 = \bigl(x_1^2 + x_2^2 + \cdots + x_d^2\bigr)^{1/2}.$
* The 1-norm, $\|\bfx\|_1 = |x_1| + |x_2| + \cdots + |x_d|.$

On the number line (i.e., $\real^1$), the distance between two values is just the absolute value of their difference, $|x-y|$. In $\real^d$, the distance between two vectors is the norm of their difference, $\| \bfx - \bfy \|$. 

The 2-norm is also called the *Euclidean norm*. It generalizes ordinary geometric distance in $\real^2$ and $\real^3$ and is usually considered the default. The 1-norm is sometimes called the *Manhattan norm*, because in $\real^2$ it represents the total number of east/west and north/south moves needed between points on a grid.

## Algorithm

As data, we are given labeled examples $\bfx_1,\ldots,\bfx_n$ in $\real^d$. Given a new query vector $\bfx$, find the $k$ labeled vectors closest to $\bfx$ and choose the most frequently occurring label among them. Ties can be broken randomly.

KNN divides up the feature space into domains that are dominated by nearby instances. The boundaries between those domains (called **decision boundaries**) can be fairly complicated, though, as shown in {numref}`fig-nn-example`.

```{figure} knn_example.png
:name: fig-nn-example
Data vectors (dots) divide feature space into classes (colors). The decision boundaries between classes are not necessarily simple. (Figure from scikit-learn.org.) 
```

Implementation of KNN is straightforward for small data sets, but requires care to get reasonable execution efficiency for large sets.  

## KNN in sklearn

Let's demonstrate cross-validation using a seaborn data set. We use `dropna` to drop any rows with missing values.
```{code-cell}
import seaborn as sns
import pandas as pd
pen = sns.load_dataset("penguins")
pen = pen.dropna()
pen
```

The data set has four quantitative columns that we use as features, and the species name is the label. 

```{code-cell}
X = pen[["bill_length_mm","bill_depth_mm","flipper_length_mm","body_mass_g"]]
y = pen["species"]
```

Scikit-learn plays nicely with pandas, so we don't have to translate the data into new structures. 

```{code-cell}
from sklearn import neighbors
knn = neighbors.KNeighborsClassifier(n_neighbors=5)
knn.fit(X,y)
```

We can manually find the neighbors of a new vector. However, we have to make the query in the form of a data frame, since that is how the training data was provided. Here we make a query frame for values very close to the ones in the first row of the data.

```{code-cell}
query = pd.DataFrame([[39,19,180,3750]],columns=X.columns)
dist,idx = knn.kneighbors(query)
idx[0]
```

As you see above, the first point (index 0) was the closest, followed by four others. We can look up the labels of these points:

```{code-cell}
y[idx[0]]
```

By a vote of 4–1, the classifier should choose Adelie as the result at this point.

```{code-cell}
knn.predict(query)
```

A quirk of the process is that some data points can be outvoted by their neighbors.

```{code-cell}
print("Predicted:")
print(knn.predict(X.loc[:5,:]))

print("\nData:")
print(y[:5])
```

To assess performance, let's apply 10-fold cross-validation to KNN learners with varying $k$.
```{code-cell}
from sklearn.model_selection import cross_val_score,KFold

K = range(1,10)
score_mean,score_std = [],[]
kf = KFold(n_splits=10,shuffle=True,random_state=1)
for k in K:
    knn = neighbors.KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn,X,y,cv=kf)
    score_mean.append(scores.mean())
    score_std.append(scores.std())

pd.DataFrame({"k":K,"accuracy mean":score_mean,"accuracy std":score_std})
```

There is no improvement here over $k=1$, in which each query just adopts the species of the nearest data vector.

The default norm in the KNN learner is the 2-norm. To use the 1-norm instead, add `metric="manhattan"` to the classifier setup call.

## Standardization

Notice that the values in the data columns of the penguin frame are scaled quite differently. In particular, the values in the body mass column are more than 20x larger than the other columns on average.

```{code-cell}
X.mean()
```

Consequently, the mass feature will dominate the distance calculations. 

To remedy this issue, we should transform the data into z-scores:

```{code-cell}
Z = X.transform( lambda x: (x-x.mean())/x.std() )
```

In this application, standardization makes performance dramatically better.

```{code-cell}
from sklearn.model_selection import cross_val_score,KFold

K = range(1,10)
score_mean,score_std = [],[]
kf = KFold(n_splits=10,shuffle=True,random_state=1)
for k in K:
    knn = neighbors.KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn,Z,y,cv=kf)
    score_mean.append(scores.mean())
    score_std.append(scores.std())

pd.DataFrame({"k":K,"accuracy mean":score_mean,"accuracy std":score_std})
```

It turns out that this is a pretty easy classification problem.