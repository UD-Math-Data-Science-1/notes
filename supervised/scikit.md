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

Let's try a classifier whose characteristics we will explain in a future section.

```{code-cell}
from sklearn import neighbors as nbr 
knn = nbr.KNeighborsClassifier(n_neighbors=11)   # specification
knn.fit(X,y)   # training
yhat = knn.predict(X)   # prediction

yhat[-6:]
```

Compared to the original labels above, so far, so good. How often is the classifier correct? We simply count up the number of correctly predicted labels and then divide by the total number of labels, $n$. 

```{code-cell}
n = len(y)
acc = sum(yhat==y)/n   # or, acc = knn.score(X,y)
print(f"accuracy is {acc:.1%}")
```

Is that good? That turns out to be a complicated question. The vast majority of loans were funded:

```{code-cell}
funded = sum(y==-1)
print(f"{funded/n:.1%} were funded")
```

Therefore, an algorithm that simply "predicted" funding every loan would do nearly as well as ours!

## Measuring binary classifier performance

To fully understand the performance of a classifier, we have to account for four cases:

* True positives (TP): Predicts "yes", actually is "yes"
* False positives (FP): Predicts "yes", actually is "no"
* True negatives (TN): Predicts "no", actually is "no"
* False negatives (FN): Predicts "no", actually is "yes"

The four cases correspond to a 2×2 table according to the states of the prediction and *ground truth*, which is the accepted correct value. The table can be filled with counts or percentages of tested instances, to create a **confusion matrix**, as illustrated in {numref}`fig-supervised-confusion`. 

```{figure} confusion.svg
---
name: fig-supervised-confusion
---
Confusion matrix
```

```{code-cell} ipython3
from sklearn import metrics
C = metrics.confusion_matrix(y,yhat,labels=[-1,1])
lbl = ["fund","reject"]
metrics.ConfusionMatrixDisplay(C,display_labels=lbl).plot();
```

Hence there are 3370 true positives (funded) and 73 true negatives (rejected). Therefore, the **accuracy** is 

$$
\newcommand{TP}{\text{TP}}
\newcommand{FP}{\text{FP}}
\newcommand{TN}{\text{TN}}
\newcommand{FN}{\text{FN}}
\text{accuracy} = \frac{\TP + \TN}{n} = \frac{3443}{4140} = 0.83164\ldots,
$$

i.e., 83.2%. However, there are four other quantities defined by putting a "number correct" value in the numerator and a sum of a row or column in the denominator:

$$
\text{recall (aka sensitvity)} &= \frac{\TP}{\TP + \FN} \\[2mm]
\text{specificity} &= \frac{\TN}{\TN + \FP} \\[2mm] 
\text{precision} &= \frac{\TP}{\TP + \FP} \\[2mm] 
\text{negative predictive value (NPV)} &= \frac{\TN}{\TN + \FN} \\ 
$$

In words, these metrics answer the following questions:

* **recall** How often are actual "yes" cases predicted correctly?
* **specificity** How often are actual "no" cases predicted correctly?
* **precision** How often are the "yes" predictions correct?
* **NPV** How often are the "no" predictions correct?

For our loan classifier, here are the scores:

```{code-cell} ipython3
TP,FN,FP,TN = C.ravel()
print(f"recall = {TP/(TP+FN):.1%}")
print(f"specificity = {TN/(TN+FP):.1%}")
print(f"precision = {TP/(TP+FP):.1%}")
print(f"NPV = {TN/(TN+FN):.1%}")
```

The recall is almost perfect: virtually nobody who should get a loan will go away disappointed. However, the low specificity would be concerning to those doing the funding, because nine in ten applicants who should be denied will be funded as well.

There are numerous ways to combine these measures into a single number other than standard accuracy. None is universally best, because different applications emphasize different aspects of performance.

### Balanced accuracy

For a binary classifier, **balanced accuracy** is the mean of recall and specificity,

$$
\frac{1}{2} \left(\frac{\TP}{\TP + \FN} + \frac{\TN}{\TN + \FP} \right).
$$

Its value is between 1/2 and 1, with larger values indicating better performance. 

```{prf:example}
Inspired by the loan example, suppose we try to sneak through a "classifier" that funds every loan. If the sample set has $k$ funded and $n-k$ declined examples, then 

$$
\TP = k,\, \TN = 0,\, \FP = n-k,\, \FN = 0.
$$

Hence the balanced accuracy is

$$
\frac{1}{2} \left(\frac{\TP}{\TP + \FN} + \frac{\TN}{\TN + \FP} \right) = \frac{1}{2}.
$$
```

For the loan example, the balanced accuracy is $0.547$, which is considerably less rosy than the accuracy measurement. Balanced accuracy counts both classes equally, no matter how frequently their samples occur. Depending on your point of view, this might be *too* harsh a judgment. After all, the always-fund classifier would get a score of 1/2 even if the data set contained just one declined loan!  

### $F_1$ score

The **$F_1$ score** of a binary classifier is the harmonic mean of the precision and the recall:

$$
\left[ \frac{1}{2} \left(\frac{\TP + \FN}{\TP} + \frac{\TP+\FP}{\TP} \right)  \right]^{-1} = \frac{2\TP}{2\TP+\FN+\FP}.
$$

This score varies between zero (poor) and one (ideal). 

The harmonic mean is the operation for combining resistors in parallel. If one of the quantities is much smaller than the other, the harmonic mean will end up close to it. Thus, $F_1$ score punishes a classifier if either recall or precision is small.

```{prf:example}
Again consider the classifier that funds every loan. Its $F_1$ score is 

$$
\TP = k,\, \TN = 0,\, \FP = n-k,\, \FN = 0.
$$

Hence the balanced accuracy is

$$
\frac{2\TP}{2\TP+\FN+\FP} = \frac{2k}{2k+n-k} = \frac{2k}{k+n}.
$$

If the fraction of funded samples is $k/n=a$, then the $F_1$ score is $a/(1+a)$, which increases smoothly from zero to one as $a$ does.
```

The loan classifier trained above has excellent recall and respectable precision, resulting in a $F_1$ score of $0.906$.

## Performance measures for multiclass classifiers

In the previous section we saw a few ways to measure the performance of a binary classifier. When there are more than two unique labels, those measures can be extended using the **one-vs-rest** paradigm. For $K$ unique labels, this paradigm poses $K$ binary questions: "Is it in class 1, or not?", "Is it in class 2, or not?", etc. This produces $K$ versions of metrics such as accuracy, recall, $F_1$-score, and so on, which can be averaged to give a single score. There are various ways to perform the averaging, depending on whether poorly represented classes are to be weighted more weakly than others. We won't give the details.

The confusion matrix also generalizes to $K$ classes. It's easiest to see how by an example. We will use a well-known data set derived from automatic recognition of handwritten digits from 0 to 9.

```{code-cell}
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn import metrics

X_tr,X_te,y_tr,y_te = train_test_split(X,y,test_size=0.25,random_state=7)

knn = neighbors.KNeighborsClassifier(n_neighbors=16)
knn.fit(X_tr,y_tr)
yhat = knn.predict(X_te)

C = metrics.confusion_matrix(y_te,yhat)
metrics.ConfusionMatrixDisplay(C).plot();
```

From the confusion matrix, we can see that, for example, the detection of "1" has 48 true positives and a total of 5 false positives. Therefore, that precision is $48/53=90.6%$. We can get all of the individual precision scores automatically.

```{code-cell}
prec = metrics.precision_score(y_te,yhat,average=None)
print([f"{p:.1%}" for p in prec])
```

To get a composite precision score, we have to specify an averaging method. The `"macro"` option simply takes the mean of the vector above.

```{code-cell}
mac = metrics.precision_score(y_te,yhat,average="macro")
print([mac,prec.mean()])
```

