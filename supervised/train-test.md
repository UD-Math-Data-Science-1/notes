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

# The train–test paradigm

Good performance of a classifier on the training set is one thing, but how will it perform on new data? This is the question of *generalization*. In order to gauge this property, we will hold back some of the labeled data from training and use it solely to test the performance.

We will continue demonstrating with the loan funding classification data set.

```{code-cell}
import numpy as np
X = np.loadtxt("data.csv",delimiter=",")
y = np.loadtxt("labels.csv",delimiter=",")
```

We use a `sckikit` helper function to help us split off a randomized 20% of the data to use for testing.

```{code-cell}
from sklearn.model_selection import train_test_split
X_tr, X_te, y_tr, y_te = train_test_split(X,y,test_size=0.2)
print(len(y_tr),"training cases and",len(y_te),"test cases")
```

Now we train on only the training data.

```{code-cell}
knn = nbr.KNeighborsClassifier(n_neighbors=1) 
knn.fit(X_tr,y_tr)  
```

If we evaluate the performance on the training data, the classifier looks perfect.

```{code-cell}
yhat = knn.predict(X_tr) 
C = metrics.confusion_matrix(y_tr,yhat,labels=[-1,1])
print(C)
```

But the picture is much different when we measure using the test set.

```{code-cell}
yhat = knn.predict(X_te) 
C = metrics.confusion_matrix(y_te,yhat,labels=[-1,1])
print(C)
```

We now see high false positive and false negative rates. This observation illustrates **overfitting**, which is the tendency of a model that performs extremely well on a training set to generalize poorly. 

## Cross-validation

Now we can try different values for the parameter in the `knn` classifier, train, and test to find the best one. Here, we use accuracy as the performance metric.

```{code-cell}
acc = []
n = len(y_te)

for k in range(1,13):
    knn = nbr.KNeighborsClassifier(n_neighbors=k) 
    knn.fit(X_tr,y_tr)  
    yhat = knn.predict(X_te) 
    agree = sum(yhat==y_te)
    acc.append(agree/n)

print(acc)
```

The experiment above suggests that we will not see much benefit from increasing the $k$ parameter past 6. 

But we have created a new problem. Suppose, as with $k$ in the example above, we have parameters we can adjust before training a classifier. (These are called *hyperparameters* in ML, because they are not parameters meant to be adjusted by the training algorithm.) If we optimize the hyperparameters based on a performance metric over the fixed test set, then we have reintroduced the possibility of overfitting; i.e., the hyperparameters can be learned from the test set, and there is nothing to check their generalization.

There are two approaches to this dilemma. One is to split the data into *three* sets for training, testing, and *validation*. The validation set is used to evaluate the winning algorithm. However, this further reduces the amount of data available for training, which is likely to hurt performance.

The other approach, called *cross-validation*, is to train the learner multiple times, each with a different split of the data into training and testing. In **$k$-fold cross-validation**, the full data set is divided into $k$ roughly equal parts called *folds*. First, the learner is trained using folds $2,3,\ldots,k$ and tested with the cases in fold 1. Then the learners are retrained using folds $1,3,\ldots,k$ and tested with the cases in fold 2. This continues until each fold has served once as the test set.

We demonstrate $k$=fold cross-validation for a particular KNN learner for $k=5$. By default, the performance metric will be the `knn.score` method, which is defined to compute accuracy.

```{code-cell}
from sklearn.model_selection import cross_val_score

knn = nbr.KNeighborsClassifier(n_neighbors=6)   # specification
knn.fit(X,y)   # training
scores = cross_val_score(knn,X,y,cv=5)

print(scores)
print("mean:",scores.mean(),"\nstd: ",scores.std())
```

It seems reasonable now to describe the accuracy of this classifier as roughly 82%.

