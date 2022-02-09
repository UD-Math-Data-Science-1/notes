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

# Regularization

As a general term, *regularization* refers to modifying something that is difficult to always compute accurately with something more tractable. For learning models, regularization is a common way to combat overfitting.

Imagine we had an $\real^{n\times 4}$ feature matrix in which the features are identical; that is, the predictor variables satisfy $x_1=x_2=x_3=x_4$, and suppose the target $y$ also equals $x_1$. Clearly, we get a perfect regression if we use

$$
y = 1x_1 + 0x_2 + 0x_3 + 0x_4.
$$

But an equally good regression is 

$$
y = \frac{1}{4}x_1 + \frac{1}{4}x_2 + \frac{1}{4}x_3 + \frac{1}{4}x_4.
$$

For that matter, so is

$$
y = 1000x_1 - 500x_2 - 500x_3 + 1x_4.
$$

A problem with more than one solution is called **ill-posed**. If we made tiny changes to the predictor variables in this thought experiment, the problem would technically be well-posed, but there would be a wide range of solutions that were very nearly correct, in which case the problem is said to be **ill conditioned**, and for practical purposes it remains just as difficult.

The ill conditioning can be regularized away by modifying the least squares loss function to penalize complexity in the model, in the form of excessively large regression coefficients. The common choices are **ridge regression**,

$$
L(\bfw) = \twonorm{ \bfX \bfw- \bfy }^2 + \alpha \twonorm{\bfw}^2,
$$

and **LASSO**, 

$$
L(\bfw) = \twonorm{ \bfX \bfw- \bfy }^2 + \alpha \onenorm{\bfw}.
$$

As $\alpha\to 0$, both forms revert to the usual least squares loss, but as $\alpha \to \infty$, the optimization becomes increasingly concerned with prioritizing a small result for $\bfw$. 

While ridge regression is an easier function to minimize quickly, LASSO has an interesting advantage, as illustrated in this figure.

```{figure} ../_static/regularization.png
```

LASSO tends to produce **sparse** results, meaning that some of the regression coefficients are zero or negligible. These zeros indicate predictor variables that have minor predictive value, which can be valuable information in itself. Moreover, when regression is run without these variables, there may be little effect on the bias, but a reduction in variance.

## Case study: Diabetes progression

We'll apply regularized regression to data collected about the progression of diabetes. 

```{code-cell}
from sklearn import datasets
diabetes = datasets.load_diabetes(as_frame=True)["frame"]
diabetes
```

First, we look at basic linear regression on all 10 predictive features in the data.

```{code-cell}
X = diabetes.iloc[:,:-1]
y = diabetes.iloc[:,-1]

from sklearn.model_selection import train_test_split

X_tr,X_te,y_tr,y_te = train_test_split(X,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_tr,y_tr)
print("linear model score:",lm.score(X_te,y_te))
```

We will find that ridge regression improves the score a bit, at least for some hyperparameter values:

```{code-cell}
from sklearn.linear_model import Ridge
rr = Ridge(alpha=0.5)
rr.fit(X_tr,y_tr)
print("ridge regression score:",rr.score(X_te,y_te))
```

A LASSO regression makes a smaller improvement:

```{code-cell}
from sklearn.linear_model import Lasso
lass = Lasso(alpha=0.2)
lass.fit(X_tr,y_tr)
print("LASSO model score:",lass.score(X_te,y_te))
```

However, while ridge regression still uses all of the features, LASSO ignores four of them:

```{code-cell}
print("ridge coeffs:")
print(rr.coef_)
print("LASSO coeffs:")
print(lass.coef_)
```

We can use the magnitude of the LASSO coefficients to rank the relative importance of the predictive features:

```{code-cell}
import numpy as np
idx = np.argsort(np.abs(lass.coef_))  # sort zero to largest
idx = idx[::-1]                       # reverse
X.columns[idx]
```

Finally, we will use cross-validation to compare basic regression with all factors, versus using just the top 5 factors:

```{code-cell}
from sklearn.model_selection import cross_val_score,KFold

kf = KFold(n_splits=8,shuffle=True,random_state=10)

scores = cross_val_score(lm,X,y,cv=kf)
print("scores with all predictors:")
print(f"mean = {scores.mean():.5f}, std = {scores.std():.4f}")

scores = cross_val_score(lm,X.iloc[:,idx[:5]],y,cv=kf)
print("scores with top 5 predictors:")
print(f"mean = {scores.mean():.5f}, std = {scores.std():.4f}")
```

When fewer factors are used, we see some reduction in variance, and the mean testing score actually goes up a bit as well.