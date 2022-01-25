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

# Logistic regression

Logistic regression is, somewhat paradoxically, most often used for classification rather than a regression problem. In the case of **binary logistic regression**, the labels of each instance is either 0 or 1, but the regressor predicts a real number between zero and one. This value is typically interpreted as the probability of observing a 1, and then a threshold is chosen to quantize the output to 0 or 1.

## Logistic and logit functions

The **logistic function** is defined as

$$
\sigma(x) = \frac{1}{1+e^{-x}}.
$$

```{figure} ../_static/logistic.png
```

The logistic function takes the form of a smoothed step up from 0 to 1. Its inverse is the **logit function**,

$$
\logit(p) = \ln\left( \frac{p}{1-p} \right).
$$

```{figure} ../_static/logit.png
```

In keeping with interpreting $p$ as probability, $\logit(p)$ is the **log-odds ratio**. For instance, if $p=2/3$, then the odds ratio is $(2/3)/(1/3)=2$ (i.e., 2:1 odds), and $\logit(2/3)=\ln(2)$. 

Logistic regression is the approximation

$$
\logit(p) \approx \bfx^T\bfw,
$$

that is, multilinear regression for the function $\logit(p)$, where $p$ is the probability of the class $y=1$. Hence

$$
p \approx \sigma(\bfx^T\bfw).
$$

## Loss function

At a training observation $(\bfx_i,y_i)$, we know that either $p=0$ or $p=1$. Let $\hat{p}_i$ be the output of the regressor at this observation:

$$
\hat{p}_i  = \sigma(\bfx_i^T\bfw).
$$

The loss function is then

$$
L(\bfw) = -\sum_{i=1}^n \left[ y_i \ln(\hat{p}_i) + (1-y_i) \ln(1-\hat{p}_i) \right].
$$

Note that observation $i$ contributes $-\ln(1-\hat{p}_i)$ if $y_i=0$ and $-\ln(\hat{p}_i)$ if $y_i=1$. Both quantities increase as $\hat{p}_i$ gets farther away from $y_i$. This loss is a special case of **cross-entropy**, a measure of dissimilarity between the probabilities of 1 occurring in the training versus the prediction.

As with other forms of linear regression, the loss function is often regularized using the ridge or LASSO penalty. As we covered earlier, there is a hyperparameter $C$ that emphasizes small $\norm{\bfw}$ as $C\to 0$, and pure regression as $C\to \infty$. 


## Case study: Personal spam filter

We will try logistic regression for a simple spam filter. The data set is based on work and personal emails for one individual. The features are calculated word and character frequencies, as well as the appearance of capital letters. 

```{code-cell}
import pandas as pd
spam = pd.read_csv("spambase.csv")
spam
```

We'll create a feature matrix and label vector, and split into train/test sets.

```{code-cell}
X = spam.drop("class",axis="columns")
y = spam["class"]

from sklearn.model_selection import train_test_split
X_tr,X_te,y_tr,y_te = train_test_split(X,y,test_size=0.2,shuffle=True,random_state=1)
```

When using norm-based regularization, it's good practice to standardize the variables, so we will prepare to set up a pipeline.

```{code-cell}
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
```

First we use a large value of $C$ to emphasize the regressive loss rather than the regularization penalty. (The default regularization norm is the 2-norm.) It's not required to select a solver, but we choose one here that is reliable for small data sets.

```{code-cell}
from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(C=100,solver="liblinear")
pipe = make_pipeline(StandardScaler(),logr)
pipe.fit(X_tr,y_tr)
pipe.score(X_te,y_te)
```

Let's look at the most extreme regression coefficients.

```{code-cell}
pd.Series(logr.coef_[0],index=X.columns).sort_values()
```

The word "george" is a strong counter-indicator for spam (remember that this data set comes from an individual), while the presence of "free" or consecutive capital letters is a strong signal of spam. 

The predictions by the regressor are all either 0 or 1. But we can also see the forecasted probabilities before rounding.

```{code-cell}
print("classes:")
print(pipe.predict(X_tr.iloc[:5,:]))
print("\nprobabilities:")
print(pipe.predict_proba(X_tr.iloc[:5,:]))
```

The probabilities might be useful, e.g., to make decisions based on the results.

For a validation-based selection of the best regularization parameter, we can use `LogisticRegressionCV`.

```{code-cell}
from sklearn.linear_model import LogisticRegressionCV
logr = LogisticRegressionCV(Cs=40,cv=5,solver="liblinear")
pipe = make_pipeline(StandardScaler(),logr)
pipe.fit(X_tr,y_tr)

print(f"best C value: {logr.C_[0]:.3g}")
print(f"R2 score: {pipe.score(X_te,y_te):.4f}")
```

## Multiclass case

When there are more than two unique labels possible, logistic regression can be extended through the **one-vs-rest** paradigm. Given $K$ classes, there are $K$ binary regressors fit for the outcomes "class 1/not class 1," "class 2/not class 2," and so on. This gives predictive relative probabilities $q_1,\ldots,q_K$ for the occurrence of each individual class. Since they need not sum to 1, they can be normalized into predicted probabilities via

$$
\hat{p}_i = \frac{q_i}{\sum_{k=1}^K q_k}.
$$

<!-- 
Another way to convert them is by using a **softmax** function:

$$
p_i = \frac{e^{q_i}}{\sum_{k=1}^K e^{q_k}}.
$$

The softmax exaggerates differences between the $q_i$, making the result closer to a "winner takes all" result.
 -->

## Case study: Gas sensor drift

As a multiclass example, we use a data set about gas sensors recording values over long periods of time.

```{code-cell}
gas = pd.read_csv("gas_drift.csv")
y = gas["Class"]
X = gas.drop("Class",axis="columns")
X_tr,X_te,y_tr,y_te = train_test_split(X,y,test_size=0.2,shuffle=True,random_state=1)

logr = LogisticRegression(solver="liblinear")
pipe = make_pipeline(StandardScaler(),logr)
pipe.fit(X_tr,y_tr)
pipe.score(X_te,y_te)
```

We can now look at predictions of probability for each class.

```{code-cell}
import pandas as pd
phat = pipe.predict_proba(X)
pd.DataFrame(phat,columns=["Class "+str(i) for i in range(1,7)])
```
