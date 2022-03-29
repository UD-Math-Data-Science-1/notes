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

# Logistic regression

The reinterpretation of classification methods as a form of regression on probability quickly leads to the question of looking for other ways to perform that regression. Specifically, can linear regression be adapted to that purpose? The answer is a qualified "yes".

A linear regressor is the function $f(\bfx) = \bfx^T \bfw$ for a constant vector $\bfw$ (where we may augment $\bfx$ with a constant in order to incorporate the intercept). It's not a good candidate for representing a probability, which should vary between 0 and 1. A simple remedy is to transform its output using the **logistic function**, which is defined as

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

A natural use of linear regression, which has a range over all real numbers, is to match it to the logit of probability, rather than to probability itself:

$$
\logit(p) \approx \bfx^T\bfw.
$$

This implies multilinear regression for the function $\logit(p)$, where $p$ is the probability of the class $y=1$. Equivalently,

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
L(\bfw) = -\sum_{i=1}^n \left[ y_i \log(\hat{p}_i) + (1-y_i) \log(1-\hat{p}_i) \right].
$$

The logarithms above can have any base, since that choice only changes $L$ by a constant factor. Note that in cross-entropy, observation $i$ contributes $-\log(1-\hat{p}_i)$ if $y_i=0$ and $-\log(\hat{p}_i)$ if $y_i=1$. This loss function creates an unboundedly large penalty as $\hat{p}_i \to 1$ if $y_i=0$, and vice versa, which often makes it preferable to the least-squares alternative above.

Logistic regression does have a major disadvantage compared to (multi)linear regression: the minimization of loss does *not* lead to a linear problem for the weight vector $\bfw$. The difference in practice is usually not concerning, though.

## Regularization

As with other forms of regression, the loss function may be regularized using the ridge or LASSO penalty. The standard formulation is 

$$
\widetilde{L}(\bfw) = C \, L(\bfw) + \norm{\bfw},
$$

where $C$ is a positive hyperparameter and the vector norm is either the 2-norm (ridge) or 1-norm (LASSO). Note that $C$ functions like the inverse of the regularization parameter $\alpha$ in our linear regressor. This is simply a different convention (like the one for the SVM), but it means that smaller values of $C$ imply *greater* amounts of regularization.

## Case study: Personal spam filter

We will try logistic regression for a simple spam filter. The data set is based on work and personal emails for one individual. The features are calculated word and character frequencies, as well as the appearance of capital letters.

```{code-cell} ipython3
import pandas as pd
spam = pd.read_csv("spambase.csv")
spam
```

We'll create a feature matrix and label vector, and split into train/test sets.

```{code-cell} ipython3
X = spam.drop("class",axis="columns")
y = spam["class"]

from sklearn.model_selection import train_test_split
X_tr,X_te,y_tr,y_te = train_test_split(X,y,test_size=0.2,shuffle=True,random_state=1)
```

When using norm-based regularization, it's good practice to standardize the variables, so we will prepare to set up a pipeline.

```{code-cell} ipython3
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
```

First we use a large value of $C$ to emphasize the regressive loss rather than the regularization penalty. (The default regularization norm is the 2-norm.) It's not required to select a solver, but we choose one here that is reliable for small data sets.

```{code-cell} ipython3
from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(C=100,solver="liblinear")
pipe = make_pipeline(StandardScaler(),logr)
pipe.fit(X_tr,y_tr)
print("accuracy:",pipe.score(X_te,y_te))
```

Let's look at the most extreme regression coefficients, associating them with the feature names and then sorting the results:

```{code-cell} ipython3
coef = pd.Series(logr.coef_[0],index=X.columns).sort_values()
print("least spammy:")
print(coef[:4])
print("\nmost spammy:")
print(coef[-4:])
```

The word "george" is a strong counter-indicator for spam (remember that this data set comes from an individual). Its presence makes the inner product $\bfx^T\bfw$ more negative, which drives the logistic function closer to 0. Conversely, the presence of consecutive capital letters increases the inner product and pushes the probability closer to 1. 

The predictions by the regressor are all either 0 or 1. But we can also see the forecasted probabilities before thresholding.

```{code-cell} ipython3
print("predicted classes:")
print(pipe.predict(X_tr.iloc[:5,:]))
print("\nprobabilities:")
print(pipe.predict_proba(X_tr.iloc[:5,:]))
```

The probabilities might be useful for making decisions based on the results. For example, the first instance above was much less certain about the classification than the second, and a lower threshold for determining spam might have changed the class to 1. The probability matrix can be used to create an ROC curve showing the tradeoffs over all thresholds.

For a validation-based selection of the best regularization parameter value, we can use `LogisticRegressionCV`, which is basically a convenience method for a grid search. You can specify which values of $C$ to search over, or just say how many, as we do here:

```{code-cell} ipython3
from sklearn.linear_model import LogisticRegressionCV
logr = LogisticRegressionCV(Cs=40,cv=5,solver="liblinear",random_state=0)
pipe = make_pipeline(StandardScaler(),logr)
pipe.fit(X_tr,y_tr)

print(f"best C value: {logr.C_[0]:.3g}")
print(f"accuracy score: {pipe.score(X_te,y_te):.4f}")
```

## Multiclass case

When there are more than two unique labels possible, logistic regression can be extended through the **one-vs-rest** (OVR) paradigm. Given $K$ classes, there are $K$ binary regressors fit for the outcomes "class 1/not class 1," "class 2/not class 2," and so on, giving $K$ different coefficient vectors, $\bfw_k$. Now for a sample point $\bfx_i$ we predict probabilities for it being in each class:

$$
\hat{q}_{i,k} = \sigma(\bfx_i^T \bfw_k), \qquad k=1,\ldots,K. 
$$

Since the $K$ OVR regressors are done independently, there is no reason to think these probabilities will sum to 1 over all the classes. But it's easy to normalize them:

$$
\hat{p}_{i,k} = \frac{\hat{q}_{i,k}}{\sum_{k=1}^K \hat{q}_{i,k}}.
$$

That is, we get a matrix of probabilities. Each of the $n$ rows gives the class probabilities at a single sample point, and each of the $K$ columns gives the probability of one class at all the samples.


## Case study: Gas sensor drift

As a multiclass example, we use a data set about gas sensors recording values over long periods of time.

```{code-cell} ipython3
gas = pd.read_csv("gas_drift.csv")
y = gas["Class"]
X = gas.drop("Class",axis="columns")
X_tr,X_te,y_tr,y_te = train_test_split(X,y,test_size=0.2,shuffle=True,random_state=1)

logr = LogisticRegression(solver="liblinear")
pipe = make_pipeline(StandardScaler(),logr)
pipe.fit(X_tr,y_tr)
print("accuracy score:",pipe.score(X_te,y_te))
```

We can now look at predictions of probability for each class.

```{code-cell} ipython3
import pandas as pd
p_hat = pipe.predict_proba(X_te)
results = pd.DataFrame(p_hat,columns=["Class "+str(i) for i in range(1,7)])
results
```

Here is a look at how the maximum prediction probability for each row in the test set is distributed:

```{code-cell} ipython3
import seaborn as sns
import numpy as np

sns.displot(x=np.max(p_hat,axis=1));
```

You can see from the plot that a solid majority of classifications are made with at least 90% probability. So if we set a high threshold for classification, we should get few false positives while still getting good recall. An AUC-ROC score can be computed by averaging the values over the curves for each class. In this case, AUC-ROC score is very high:

```{code-cell} ipython3
from sklearn.metrics import roc_auc_score
roc_auc_score(y_te,p_hat,multi_class="ovr")
```

<div style="max-width:608px"><div style="position:relative;padding-bottom:66.118421052632%"><iframe id="kaltura_player" src="https://cdnapisec.kaltura.com/p/2358381/sp/235838100/embedIframeJs/uiconf_id/43030021/partner_id/2358381?iframeembed=true&playerId=kaltura_player&entry_id=1_mr2gh70i&flashvars[streamerType]=auto&amp;flashvars[localizationCode]=en&amp;flashvars[leadWithHTML5]=true&amp;flashvars[sideBarContainer.plugin]=true&amp;flashvars[sideBarContainer.position]=left&amp;flashvars[sideBarContainer.clickToClose]=true&amp;flashvars[chapters.plugin]=true&amp;flashvars[chapters.layout]=vertical&amp;flashvars[chapters.thumbnailRotator]=false&amp;flashvars[streamSelector.plugin]=true&amp;flashvars[EmbedPlayer.SpinnerTarget]=videoHolder&amp;flashvars[dualScreen.plugin]=true&amp;flashvars[Kaltura.addCrossoriginToIframe]=true&amp;&wid=1_en335xwu" width="608" height="402" allowfullscreen webkitallowfullscreen mozAllowFullScreen allow="autoplay *; fullscreen *; encrypted-media *" sandbox="allow-forms allow-same-origin allow-scripts allow-top-navigation allow-pointer-lock allow-popups allow-modals allow-orientation-lock allow-popups-to-escape-sandbox allow-presentation allow-top-navigation-by-user-activation" frameborder="0" title="Kaltura Player" style="position:absolute;top:0;left:0;width:100%;height:100%"></iframe></div></div>
