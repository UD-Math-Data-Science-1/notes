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

# Using scikit-learn

As a data set, we use loan applications on a crowdfunding site. Each feature vector is of length 15, indicating factors such as the amount requested, the interest rate, the applicant's annual income, etc. The goal is to predict whether the loan is at least partly funded.

In order to work with vectors and matrices, which are both types of arrays, we will use the `numpy` package.

```{code-cell} ipython3
import numpy as np
X = np.loadtxt("data.csv",delimiter=",")
print("feature matrix has shape",X.shape)
y = np.loadtxt("labels.csv",delimiter=",")
print("label vector has shape",y.shape)
n,d = X.shape
print("there are $d features and $n samples")
```

Let's look at the first 5 features of the first instance.

```{code-cell}
X[0,:5]
```

And here are the last 6 labels.

```{code-cell}
y[-6:]
```

A label $-1$ indicates that the loan was funded, while $1$ indicates that it was rejected.

We will use the scikit-learn (`sklearn`) package to get familiar with classifiers. There are three main activities in this package:

* **fit**, to train the classifier
* **predict**, to apply the classifier
* **transform**, to modify the data

We'll explore fitting and prediction for now. Let's try a classifier whose characteristics we will explain in a future section.

```{code-cell}
from sklearn import neighbors 
knn = neighbors.KNeighborsClassifier(n_neighbors=11)   # specification
knn.fit(X,y)            # training
```

At this point, the classifier object `knn` has figured out what it needs to do with the training data, and we can ask it to make predictions. Each application we want a prediction for is a vector with 15 components (features). The prediction query has to be a 2D array, with each row being a query vector. The result is a vector of predictions.

```{code-cell}
Xq = 100*np.ones((1,d))
knn.predict(Xq)
```

We don't have any realistic application data at hand, other than the training data. By comparing the predictions made for that data to the true labels we supplied, we can try to get an idea of how accurate the predictor is.

```{code-cell}
yhat = knn.predict(X)   # prediction
yhat[-6:]
```

Compared to the true labels we printed out above, so far, so good. Now simply count up the number of correctly predicted labels and then divide by the total number of labels, $n$. 

```{code-cell}
acc = sum(yhat==y)/n 
print(f"accuracy is {acc:.1%}")
```

Of course, scikit has functions for doing all this in fewer steps.

```{code-cell}
from sklearn import metrics

acc = metrics.accuracy_score(y,yhat)
print(f"accuracy is {acc:.1%}")

acc = knn.score(X,y)
print(f"accuracy is {acc:.1%}")
```

## Train–test paradigm

Good performance of a classifier on the samples used to train seems to be necessary, but is it sufficient? We are more interested on how the classifier performs on new data. This is the question of *generalization*. In order to gauge generalization, we hold back some of the labeled data from training and use it only to test the performance.


<!-- 
```{code-cell}
import numpy as np
X = np.loadtxt("data.csv",delimiter=",")
y = np.loadtxt("labels.csv",delimiter=",")
``` 
-->

A `sklearn` helper function allows us to split off a randomized 20% of the data to use for testing:

```{code-cell}
from sklearn.model_selection import train_test_split
X_tr, X_te, y_tr, y_te = train_test_split(X,y,test_size=0.2)
print(len(y_tr),"training cases and",len(y_te),"test cases")
```

Now we train on the training data...

```{code-cell}
from sklearn import neighbors as nbr
knn = nbr.KNeighborsClassifier(n_neighbors=1)
knn.fit(X_tr,y_tr)
```

...and test on the rest.

```{code-cell}
acc = knn.score(X_te,y_te)
print(f"accuracy is {acc:.1%}")
```

This is a less flattering result than when we computed accuracy on the full set.

## sklearn and pandas

Scikit-learn plays very nicely with pandas. You can use data frames for the sample values and labels. Let's look at a data set that comes from seaborn.

```{code-cell}
import seaborn as sns
penguins = sns.load_dataset("penguins")
penguins = penguins.dropna()
penguins
```

The `dropna` call above removes rows that have a `NaN` value in any column, as sklearn doesn't handle those well all the time. This data frame has four quantitative columns that we will use as features. We will use the species column for the labels. 

```{code-cell}
X = penguins[["bill_length_mm","bill_depth_mm","flipper_length_mm","body_mass_g"]]
y = penguins["species"]
```

Now `X` is a data frame and `y` is a series. They can be input directly into a learning method call.

```{code-cell}
knn = neighbors.KNeighborsClassifier(n_neighbors=5)
knn.fit(X,y);
```

To make predictions, we need to pass in another data frame that has the same columns as `X`. Here is a simple way to turn a numerical vector or list into such a frame.

```{code-cell}
import pandas as pd
x_new = [39,19,180,3750]
xdf = pd.DataFrame([x_new],columns=X.columns)
xdf
```

(The `[x_new]` part does need to have the brackets, so that pandas sees a list of row values there. We could put multiple rows in that list.) Now we can use the classifier to make a prediction.

```{code-cell}
knn.predict(xdf)
```

The result comes back as a series of the same dtype as `y`.

This may seem like hassle and extra work for mere window dressing. Why not just work with arrays? When we are focusing on the math, that's fine. But in an application, it's easy to lose track of what the integer indexes of an array are supposed to mean. By using their names, you keep things clearer in your own mind, and scikit will give you warnings and errors if you aren't using them consistently. In other words, it's *productive* hassle.

