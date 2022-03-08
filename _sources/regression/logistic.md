---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: 'Python 3'
  language: python
  name: python3
---

# Logistic regression

The reinterpretation of classification methods as a form of regression on probability quickly leads to the question of looking for other ways to perform that regression. Specifically, can linear regression be adapted to that purpose? 

A linear regressor is the function $f(\bfx) = \bfx^T \bfw$ for a constant vector $\bfw$ (where we may augment $\bfx$ with a constant in order to incorporate the intercept). It's not a good candidate for representing a probability, which should vary between 0 and 1. A simple remedy is to transform its output using **logistic function**, which is defined as

$$
\sigma(x) = \frac{1}{1+e^{-x}}.
$$

```{figure} ../_static/logistic.png
```

The logistic function has the real line as its domain and takes the form of a smoothed step up from 0 to 1. Its inverse is the **logit function**,

$$
\logit(p) = \ln\left( \frac{p}{1-p} \right).
$$

```{figure} ../_static/logit.png
```

When interpreting $p$ as probability, $\logit(p)$ is the **log-odds ratio**. For instance, if $p=2/3$, then the odds ratio is $(2/3)/(1/3)=2$ (i.e., 2:1 odds), and $\logit(2/3)=\ln(2)$. 

The logical use of linear regression, which has an unbounded range, is to match that to the logit of probability, rather than to probability itself:

$$
\logit(p) \approx \bfx^T\bfw,
$$

that is, multilinear regression for the function $\logit(p)$, where $p$ is the probability of the class $y=1$. Equivalently,

$$
p \approx \sigma(\bfx^T\bfw).
$$

The resulting method is called **logistic regression**.

## Loss function

At each training observation $(\bfx_i,y_i)$, we know that either $y_i=0$ or $y_i=1$. Extending the loss function for linear regression to the logistic case would suggest the minimization of least squares,

$$
\sum_{i=1}^n \left[ \bfx_i^T\bfw - \logit(y_i) \right]^2. 
$$

However, the logits in this expression are all infinite, so a different loss function must be identified. One possibility is 

$$
\sum_{i=1}^n \left[ \hat{p}_i - y_i \right]^2, \qquad \hat{p}_i = \sigma(\bfx_i^T\bfw) .
$$

It's more common to minimize the **cross-entropy** loss function

$$
L(\bfw) = -\sum_{i=1}^n \left[ y_i \ln(\hat{p}_i) + (1-y_i) \ln(1-\hat{p}_i) \right].
$$

Note that observation $i$ contributes $-\ln(1-\hat{p}_i)$ if $y_i=0$ and $-\ln(\hat{p}_i)$ if $y_i=1$. This loss function creates an unboundedly large penalty as $\hat{p}_i \to 1$ if $y_i=0$, and vice versa. 

Logistic regression has a major disadvantage compared to multilinear regression: the minimization of loss does *not* lead to a linear problem for the weight vector $\bfw$. The difference in practice is usually not concerning, though. As with other forms of regression, the loss function may be regularized using the ridge or LASSO penalty. As we covered earlier, there is a regularization parameter $C$ that emphasizes small $\norm{\bfw}$ as $C\to 0$, and pure regression as $C\to \infty$. 


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
