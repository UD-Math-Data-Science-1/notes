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

# Learning performance


## Overfitting

If we evaluate the classifier's performance on the training data, the classifier looks perfect by any metric:

```{code-cell}
from sklearn import metrics
yhat = knn.predict(X_tr)
C = metrics.confusion_matrix(y_tr,yhat,labels=[-1,1])
print(C)
```

But the picture is much different when we measure on the test set.

```{code-cell}
yhat = knn.predict(X_te)
C = metrics.confusion_matrix(y_te,yhat,labels=[-1,1])
print(C)
```

We now see high false positive and false negative rates. This observation illustrates **overfitting**, which is the poor generalization of a model that learns too many idiosyncratic details about a training set.

## Bias–variance tradeoff

Suppose that $f(x)$ is a function giving the ground truth over the entire population. Let $\hat{f}(x)$ denote a learning algorithm obtained after training. Conceptually, $\hat{f}$ is just one realization that we get from one particular training set. We use $E[\cdot]$ to denote the process of averaging over all possible training sets. Note that the average of a sum of quantities is the sum of the averages.

Hence, the mean error is

$$
E\bigl[ f(x) - \hat{f}(x) \bigr] = f(x) - E\bigl[ \hat{f}(x) \bigr] = y - \hat{y},
$$

where $y=f(x)$ and $\hat{y}$ is the average prediction. This quantity is called the **bias**. 

Now we look at mean squared error:

$$
E\bigl[ (f(x) - \hat{f}(x))^2 \bigr] &= E\bigl[ (f(x) - \hat{y} + \hat{y} - \hat{f}(x))^2 \bigr] \\ 
&= E\bigl[ (f(x) - \hat{y})^2 \bigr] + E\bigl[ (\hat{y} - \hat{f}(x))^2 \bigr] - 2E\bigl[ (f(x) - \hat{y})(\hat{y} - \hat{f}(x)) \bigr] \\ 
&= (f(x) - \hat{y})^2 + E\bigl[ (\hat{y} - \hat{f}(x))^2 \bigr] - 2(f(x) - \hat{y}) E\bigl[ \hat{y} - \hat{f}(x) \bigr] \\
&= (f(x) - \hat{y})^2 + E\bigl[ (\hat{y} - \hat{f}(x))^2 \bigr].
$$

The first term is the squared bias. The second is the **variance** of the learning method. In words,

* Bias: How close is $\hat{y}$ to $f(x)$? 
* Variance: How close to $\hat{y}$ is our $\hat{f}(x)$ likely to be?

There is a crude analogy with hitting the bullseye on a dartboard. A low-variance, high-bias learner will throw a tight cluster of darts far from the bullseye. A low-bias, high-variance learner will scatter the darts evenly all over the board. When learners are overfitted, their output on a test set depends sensitively on the choice of training set, which creates a large variance.

Continuing with the loan data, we can train on multiple subsets of the full data set in order to simulate the effect of the training set.

```{code-cell}
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import seaborn as sns
import pandas as pd

X_tr, X_te, y_tr, y_te = train_test_split(X,y,test_size=0.2)
n = 1000             # size of the training subset
err,kind = [],[]    # track information for the result
knn = nbr.KNeighborsClassifier(n_neighbors=2)
for i in range(200):
    X_tr,y_tr = shuffle(X_tr,y_tr,random_state=1)
    knn.fit(X_tr[:n,:],y_tr[:n])   # training
    err.append(1-knn.score(X_tr[:n,:],y_tr[:n]))
    err.append(1-knn.score(X_te,y_te))
    kind.extend(["train","test"])

result = pd.DataFrame({"error":err,"kind":kind})
sns.displot(data=result,x="error",hue="kind",bins=24);
```

The histograms above show that the error measurements based on the training set are usually substantially less than on the test set. This likely indicates overfitting. We also see a lot of variance in the test set measurements. 

Most learning algorithms have one or more **hyperparameters** that are selected in advance by the designer rather than adjusted to fit the training data. Often, a hyperparameter adjusts the number of degrees of freedom available to use for fitting. Giving the learner more power might lead to decreased bias, because there is a larger universe of potential classifiers to choose from. But it tends to increase variance, because the higher fidelity is actually used to fit more closely to the particular training set that is chosen. This dilemma is generally known as the **bias–variance tradeoff**.




## Learning curves

Let's illustrate overfitting with *decision trees*, a different type of classifier to be studied later. A decision tree has a controllable maximum depth which, when increased, allows it more freedom to fit training data.

For this experiment, we vary $n$, the number of instances used to train the classifier. We first split off part of the data to serve as a test set. For each $n$, the rest of the data is randomly reordered before training, and we measure performance on the training set as well as the test set. As performance metric we use the `score` method of the classifier to get its accuracy, then subtract it from 1 to get an error measurement.

```{code-cell}
from sklearn import tree
from sklearn.utils import shuffle
import seaborn as sns
import pandas as pd

X_tr, X_te, y_tr, y_te = train_test_split(X,y,test_size=0.2)
N = range(50,3201,20)
train_err = []
test_err = []
for n in N:
  X_tr,y_tr = shuffle(X_tr,y_tr,random_state=1)
  knn = tree.DecisionTreeClassifier(max_depth=5)   # specification
  knn.fit(X_tr[:n,:],y_tr[:n])   # training
  train_err.append(1-knn.score(X_tr[:n,:],y_tr[:n]))
  test_err.append(1-knn.score(X_te,y_te))

result = pd.DataFrame(
    {"train error":train_err,"test error":test_err},
    index=pd.Series(N,name="size of training set")
)
sns.lineplot(data=result)
```

The plot above shows **learning curves**. Both curves converge to a horizontal asymptote. The gap between the curves is due to variance, which decreases as the training set grows. This is to be expected; generalizing from a few examples is probably harder than when many are available. The height of the curves is due to bias, which appears to be somewhere around a 16% error rate. This is a lower bound on the actual error, regardless of the training set; you can't knock out an elephant with a feather, no matter how many times you whack her with it.

The curves above were for a tree depth of 5. The next plot shows it for a depth of 10, which increases the approximation power.

```{code-cell}
---
tags: [hide-input]
---
X_tr, X_te, y_tr, y_te = train_test_split(X,y,test_size=0.2)
N = range(50,3201,20)
train_err = []
test_err = []
for n in N:
  X_tr,y_tr = shuffle(X_tr,y_tr,random_state=1)
  knn = tree.DecisionTreeClassifier(max_depth=10)   # specification
  knn.fit(X_tr[:n,:],y_tr[:n])   # training
  train_err.append(1-knn.score(X_tr[:n,:],y_tr[:n]))
  test_err.append(1-knn.score(X_te,y_te))

result = pd.DataFrame(
    {"train error":train_err,"test error":test_err},
    index=pd.Series(N,name="size of training set")
)
sns.lineplot(data=result)
```

This time, the curves do not come together before we run out of data. The nonzero variance suggests that the learner is in some sense overqualified to fit the available data. The bias did not get below 17% and appears to be levelling off, so that increased approximation power isn't even useful.

Finally, we see what happens with a smaller depth of just 2.

```{code-cell}
---
tags: [hide-input]
---
X_tr, X_te, y_tr, y_te = train_test_split(X,y,test_size=0.2)
N = range(50,3201,20)
train_err = []
test_err = []
for n in N:
  X_tr,y_tr = shuffle(X_tr,y_tr,random_state=1)
  knn = tree.DecisionTreeClassifier(max_depth=10)   # specification
  knn.fit(X_tr[:n,:],y_tr[:n])   # training
  train_err.append(1-knn.score(X_tr[:n,:],y_tr[:n]))
  test_err.append(1-knn.score(X_te,y_te))

result = pd.DataFrame(
    {"train error":train_err,"test error":test_err},
    index=pd.Series(N,name="size of training set")
)
sns.lineplot(data=result)
```

Now the variance is zero almost from the start. The bias is around 17%, so we have paid a small price for the lost power. The ideal bias–variance tradeoff in this case is probably a depth of 4 or 5. 

## Cross-validation

We would like to use a performance metric over a test set to choose hyperparameters optimally. However, if we base the hyperparameter optimization on a fixed test set, then we are effectively learning from that set! That is, the hyperparameters might become too tuned to our particular choice of the test set, creating variance. To avoid this situation, we will use **cross-validation**, in which each learner is trained multiple times, each time using a different train–test split of the data. 

In **$k$-fold cross-validation**, the full data set is divided into $k$ roughly equal parts called *folds*. First, the learner is trained using folds $2,3,\ldots,k$ and tested against the cases in fold 1. Then the learner is retrained using folds $1,3,\ldots,k$ and tested against the cases in fold 2. This continues until each fold has served once as the test set.

Here we use 10-fold cross-validation for KNN learners with varying $k$. By default, the performance metric will be the `knn.score` method, which is defined to compute accuracy.

```{code-cell}
from sklearn.model_selection import cross_val_score,KFold

K = range(1,9)
acc = []
kf = KFold(n_splits=5,shuffle=True,random_state=1)
for k in K:
    knn = nbr.KNeighborsClassifier(n_neighbors=k) 
    scores = cross_val_score(knn,X,y,cv=kf)
    acc.append(scores.mean())

for (k,a) in zip(K,acc):
    print("k =",k,":",f"{a:.2%}")
```

There is little in the results to support going beyond $k=2$ for this classifier type.

