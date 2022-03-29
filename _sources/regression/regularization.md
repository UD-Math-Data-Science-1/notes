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

# Regularization

As a general term, *regularization* refers to modifying something that is difficult to compute accurately with something more tractable. For learning models, regularization is a common way to combat overfitting.

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

```{code-cell} ipython3
from sklearn import datasets
diabetes = datasets.load_diabetes(as_frame=True)["frame"]
diabetes
```

First, we look at basic linear regression on all 10 predictive features in the data.

```{code-cell} ipython3
X = diabetes.drop("target",axis=1)
y = diabetes["target"]

from sklearn.model_selection import train_test_split
X_tr,X_te,y_tr,y_te = train_test_split(X,y,test_size=0.2,random_state=2)

from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_tr,y_tr)
print("linear model CoD score:",lm.score(X_te,y_te))
```

We will find that ridge regression improves the score a bit, at least for some hyperparameter values:

```{code-cell} ipython3
from sklearn.linear_model import Ridge
rr = Ridge(alpha=0.1)  # alpha weights the regularization term
rr.fit(X_tr,y_tr)
print("ridge regression CoD score:",rr.score(X_te,y_te))
```

Ridge regularization added a penalty for the 2-norm of the regression coefficients vector. Accordingly, the regularized solution has smaller coefficients:

```{code-cell} ipython3
from numpy.linalg import norm
print("2-norm of unregularized model coefficients:",norm(lm.coef_))
print("2-norm of ridge regression coefficients:",norm(rr.coef_))
```

As we continue to increase the regularization parameter, the method becomes increasingly obsessed with keeping the coefficient vector small, and pays ever less attention to the data as a result. Eventually, the quality of fit will decrease.

```{code-cell} ipython3
rr = Ridge(alpha=4)  # more regularization
rr.fit(X_tr,y_tr)
print("2-norm of coefficient vector:",norm(rr.coef_))
print("ridge regression CoD score:",rr.score(X_te,y_te))
```

LASSO penalizes the 1-norm of the coefficient vector. Here's a LASSO regression fit:

```{code-cell} ipython3
from sklearn.linear_model import Lasso
lass = Lasso(alpha=0.05)
lass.fit(X_tr,y_tr)
print("LASSO model CoD score:",lass.score(X_te,y_te))
print("1-norm of LASSO coefficient vector:",norm(lass.coef_,1))
print("1-norm of unregularized coefficient vector:",norm(lm.coef_,1))
```

A validation curve suggests modest gains in the $R^2$ score as the regularization parameter is varied:

```{code-cell} ipython3
from sklearn.model_selection import KFold,validation_curve
import numpy as np
kf = KFold(n_splits=4,shuffle=True,random_state=0)

alpha = np.linspace(0,0.1,80)[1:]  # exclude alpha=0
_,scores = validation_curve(lass,X_tr,y_tr,cv=kf,param_name="alpha",param_range=alpha)
```

```{code-cell} ipython3
import seaborn as sns
sns.relplot(x=alpha,y=np.mean(scores,axis=1));
```

However, while ridge regression still uses all of the features, LASSO ignores four of them:

```{code-cell} ipython3
print("ridge coeffs:")
print(rr.coef_)
lass = Lasso(alpha=0.05)
lass.fit(X_tr,y_tr)
print("LASSO coeffs:")
print(lass.coef_)
```

We can use the magnitude of the LASSO coefficients to rank the relative importance of the predictive features. We have to make sure to take the absolute values of the coefficients, because we don't care about whether an effect is positive or negative, just its magnitude.

```{code-cell} ipython3
import numpy as np
# Get the permutation that sorts values in increasing order.
idx = np.argsort(np.abs(lass.coef_))  
idx = idx[::-1]    # reverse the order
idx
```

The last three features were dropped by LASSO:

```{code-cell} ipython3
zeroed = X.columns[idx[-3:]]
print(zeroed)
```

Now we can drop these features from the dataset:

```{code-cell} ipython3
X_tr_reduced = X_tr.drop(zeroed,axis=1)
X_te_reduced = X_te.drop(zeroed,axis=1)
X_tr_reduced.head(5)
```

Returning to the original, unregularized fit, we find that hardly anything is lost by using the reduced feature set:

```{code-cell} ipython3
print("original linear model score:",lm.score(X_te,y_te))
lm.fit(X_tr_reduced,y_tr)
print("reduced linear model score:",lm.score(X_te_reduced,y_te))
```

<div style="max-width:608px"><div style="position:relative;padding-bottom:66.118421052632%"><iframe id="kaltura_player" src="https://cdnapisec.kaltura.com/p/2358381/sp/235838100/embedIframeJs/uiconf_id/43030021/partner_id/2358381?iframeembed=true&playerId=kaltura_player&entry_id=1_irlwjqis&flashvars[streamerType]=auto&amp;flashvars[localizationCode]=en&amp;flashvars[leadWithHTML5]=true&amp;flashvars[sideBarContainer.plugin]=true&amp;flashvars[sideBarContainer.position]=left&amp;flashvars[sideBarContainer.clickToClose]=true&amp;flashvars[chapters.plugin]=true&amp;flashvars[chapters.layout]=vertical&amp;flashvars[chapters.thumbnailRotator]=false&amp;flashvars[streamSelector.plugin]=true&amp;flashvars[EmbedPlayer.SpinnerTarget]=videoHolder&amp;flashvars[dualScreen.plugin]=true&amp;flashvars[Kaltura.addCrossoriginToIframe]=true&amp;&wid=1_zgo9xrkv" width="608" height="402" allowfullscreen webkitallowfullscreen mozAllowFullScreen allow="autoplay *; fullscreen *; encrypted-media *" sandbox="allow-forms allow-same-origin allow-scripts allow-top-navigation allow-pointer-lock allow-popups allow-modals allow-orientation-lock allow-popups-to-escape-sandbox allow-presentation allow-top-navigation-by-user-activation" frameborder="0" title="Kaltura Player" style="position:absolute;top:0;left:0;width:100%;height:100%"></iframe></div></div>
