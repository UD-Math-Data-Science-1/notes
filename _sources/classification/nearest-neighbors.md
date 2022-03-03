---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: 'Python 3.8.8 64-bit (''base'': conda)'
  language: python
  name: python3
---
# Nearest neighbors

Our first learning algorithm is conceptually simple: Given a new point to classify, survey the nearest examples and choose the most frequent class. This is called the **$k$ nearest neighbors** (KNN) algorithm, where $k$ is the number of neighboring examples to survey.

## Norms

The existence of "closest" examples means that we need to define a notion of distance in spaces of any dimension. Let $\real^d$ be the space of vectors with $d$ real components, and let $\bfzero$ be the vector of all zeros.

```{prf:definition}
A **norm** is a function $\norm{\bfx}$ on $\real^d$ that satisfies the following properties:

$$
\norm{\bfzero} &= 0,  \\ 
\norm{\bfx} &> 0 \text{ if $\bfx$ is a nonzero vector}, \\ 
\norm{c\bfx} &= |c| \, \norm{x} \text{ for any real number $c$}, \\ 
\norm{\bfx + \bfy } &\le \norm{\bfx} + \norm{\bfy} 
$$
```

The last inequality above is called the **triangle inequality**. It turns out that these four characteristics are all we expect from a function that behaves like a distance. 

On the number line (i.e., $\real^1$), the distance between two values is just the absolute value of their difference, $|x-y|$. In $\real^d$, the distance between two vectors is the norm of their difference, $\norm{ \bfx - \bfy }$. 

We will 0encounter two different norms:

* The 2-norm, $\twonorm{\bfx} = \bigl(x_1^2 + x_2^2 + \cdots + x_d^2\bigr)^{1/2}.$
* The 1-norm, $\onenorm{\bfx} = |x_1| + |x_2| + \cdots + |x_d|.$


The 2-norm is also called the *Euclidean norm*. It generalizes ordinary geometric distance in $\real^2$ and $\real^3$ and is usually considered the default. The 1-norm is sometimes called the *Manhattan norm*, because in $\real^2$ it represents the total number of east/west and north/south moves needed between points on a grid.

## Algorithm

As data, we are given labeled examples $\bfx_1,\ldots,\bfx_n$ in $\real^d$. Given a new query vector $\bfx$, find the $k$ labeled vectors closest to $\bfx$ and choose the most frequently occurring label among them. Ties can be broken randomly.

KNN divides up the feature space into domains that are dominated by nearby instances. The boundaries between those domains (called **decision boundaries**) can be fairly complicated, though, as shown in the animation below. 

```{raw} html
<video width=640 controls src="../_static/knn_demo.mp4"></video>
```

Implementation of KNN is straightforward for small data sets, but requires care to get reasonable execution efficiency for large sets.

## KNN in sklearn

Let's revisit the penguins. We use `dropna` to drop any rows with missing values.
```{code-cell}
import seaborn as sns
import pandas as pd
penguins = sns.load_dataset("penguins")
penguins = penguins.dropna()
penguins
```

The data set has four quantitative columns that we use as features, and the species name is the label. 

```{code-cell}
col = ["bill_length_mm","bill_depth_mm","flipper_length_mm","body_mass_g"]
X = penguins[col]
y = penguins["species"]
```

Scikit-learn plays nicely with pandas, so we don't have to translate the data into new structures. 

```{code-cell}
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
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

By a vote of 4–1, the classifier should choose Adelie as the result at this location.

```{code-cell}
knn.predict(query)
```

Note that some data points can be outvoted by their neighbors.

```{code-cell}
print("Predicted:")
print(knn.predict(X.loc[:5,:]))

print("\nData:")
print(y[:5].values)
```

Here we split into training and test sets to gauge the performance of the classifier The `classification_report` function creates a summary of some of the important metrics.

```{code-cell}
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix

X_tr, X_te, y_tr, y_te = train_test_split(X,y,test_size=0.2)
knn.fit(X_tr,y_tr)

yhat = knn.predict(X_te)
print(confusion_matrix(y_te,yhat))
print(classification_report(y_te,yhat))
```

The default norm in the KNN learner is the 2-norm. To use the 1-norm instead, add `metric="manhattan"` to the classifier construction call.

## Standardization

The values in the columns of the penguin frame are scaled quite differently. In particular, the values in the body mass column are more than 20x larger than the other columns on average. Consequently, the mass feature will dominate the distance calculations.

```{code-cell}
X.mean()
```

To remedy this issue, we should transform the data into z-scores:

```{code-cell}
Z = X.transform( lambda x: (x-x.mean())/x.std() )
```

In this application, standardization makes performance dramatically better.

```{code-cell}
Z_tr, Z_te, y_tr, y_te = train_test_split(Z,y,test_size=0.2)
knn.fit(Z_tr,y_tr)

yhat = knn.predict(Z_te)
print(confusion_matrix(y_te,yhat))
print(classification_report(y_te,yhat))
```

## Pipelines

One inconvenience of the standardization step above is that it must be performed for any new data vector that comes along. Moreover, that standardization has to use the mean and std from our original creation of `Z`, so those values need to be tracked. 

The sklearn answer to this need is to create a **pipeline** that includes the transformation. Pipelines make it fairly easy to chain together data transformations, followed by a learner. The composite object acts like the original learner.

As you might guess, standardization of data is so common that it is predefined.

```{code-cell}
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler   # converts to z-scores

X_tr, X_te, y_tr, y_te = train_test_split(X,y,test_size=0.2)

knn = KNeighborsClassifier(n_neighbors=5)
pipe = make_pipeline(StandardScaler(),knn)

pipe.fit(X_tr,y_tr)
pipe.score(X_te,y_te)
```

We can look under the hood a bit. The mean and variance of each of the original data columns is stored in the first part of the pipeline.

```{code-cell}
print(pipe[0].mean_)
print(pipe[0].var_)
```

<div style="max-width:608px"><div style="position:relative;padding-bottom:66.118421052632%"><iframe id="kaltura_player" src="https://cdnapisec.kaltura.com/p/2358381/sp/235838100/embedIframeJs/uiconf_id/43030021/partner_id/2358381?iframeembed=true&playerId=kaltura_player&entry_id=1_wghs045x&flashvars[streamerType]=auto&amp;flashvars[localizationCode]=en&amp;flashvars[leadWithHTML5]=true&amp;flashvars[sideBarContainer.plugin]=true&amp;flashvars[sideBarContainer.position]=left&amp;flashvars[sideBarContainer.clickToClose]=true&amp;flashvars[chapters.plugin]=true&amp;flashvars[chapters.layout]=vertical&amp;flashvars[chapters.thumbnailRotator]=false&amp;flashvars[streamSelector.plugin]=true&amp;flashvars[EmbedPlayer.SpinnerTarget]=videoHolder&amp;flashvars[dualScreen.plugin]=true&amp;flashvars[Kaltura.addCrossoriginToIframe]=true&amp;&wid=1_vm7jak75" width="608" height="402" allowfullscreen webkitallowfullscreen mozAllowFullScreen allow="autoplay *; fullscreen *; encrypted-media *" sandbox="allow-forms allow-same-origin allow-scripts allow-top-navigation allow-pointer-lock allow-popups allow-modals allow-orientation-lock allow-popups-to-escape-sandbox allow-presentation allow-top-navigation-by-user-activation" frameborder="0" title="Kaltura Player" style="position:absolute;top:0;left:0;width:100%;height:100%"></iframe></div></div>
