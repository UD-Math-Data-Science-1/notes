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

# Multilinear and polynomial regression

We can extend linear regression to $d$ predictor variables $x_1,\ldots,x_d$:

$$
y \approx f(\bfx) = w_1 x_1 + w_2x_2 + \cdots w_d x_d + b.
$$

We can drop the intercept term $b$ from the discussion, because we could always define an additional constant predictor variable $x_{d+1}=1$ and get the same effect. So

$$
y \approx f(\bfx) = w_1 x_1 + w_2x_2 + \cdots w_d x_d = \bfw^T\bfx = \bfx^T\bfw,
$$

where we have introduced inner product notation. This is called **multilinear regression**, although it could also be called just *linear regression* in many contexts.

To create the least squares loss function, we use $\bfx_i$ to denote the $i$th row of the $n\times d$ feature matrix $\bfX$. Then  

$$
L(\bfw) = \sum_{i=1}^n (f(\bfx_i)-y_i)^2 = \sum_{i=1}^n (\bfx_i^T\bfw- y_i)^2.
$$

Introducing the shorthand notation (standard linear algebra)

$$
\bfX \bfw = \begin{bmatrix} \bfx_1^T\bfw \\ \bfx_2^T\bfw \\ \vdots \\ \bfx_n^T\bfw \end{bmatrix},
$$

we now get the compact expression

$$
L(\bfw) = \twonorm{\bfX \bfw- \bfy}^2.
$$

As in the univariate case, minimizing the loss boils down to solving a linear system of equations, known as the *normal equations*, for $\bfw$.

## Case study: Advertising and sales

Here we load data about advertising spending on different media in many markets.

```{code-cell} ipython3
import pandas as pd
ads = pd.read_csv("advertising.csv")
ads
```

Pairwise scatter plots yield some hints about what to expect from this dataset:

```{code-cell} ipython3
import seaborn as sns
sns.pairplot(ads);
```

The three types of media spending have about the same order of magnitude. The clearest association between *Sales* and spending is with *TV*. So we first try a univariate linear fit of sales against TV spending alone.

```{code-cell} ipython3
X = ads.drop("Sales",axis="columns")
X_tv = ads[["TV"]]    # has to be a frame, so ["TV"] not "TV"
y = ads["Sales"]

from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_tv,y)
print("R^2 score:",f"{lm.score(X_tv,y):.4f}")
print("Model coeffs:",lm.coef_)
```

The coefficient of determination is quite good. Next we try folding in *Newspaper* as well:

```{code-cell} ipython3
X_tv_news = ads[["TV","Newspaper"]]
lm.fit(X_tv_news,y)
print("R^2 score:",f"{lm.score(X_tv_news,y):.4f}")
print("Model coeffs:",lm.coef_)
```

This additional feature had very little effect on the quality of fit. We go on to fit using all three features:

```{code-cell} ipython3
X = ads[["TV","Newspaper","Radio"]]
lm.fit(X,y)
print("R^2 score:",f"{lm.score(X,y):.4f}")
print("Model coeffs:")
print(pd.Series(lm.coef_,index=X.columns))
```

Judging by the coefficients of the model, it's even clearer now that we can explain *Sales* very well without contributions from *Newspaper*. In order to reduce model variance, it would be reasonable to leave that column out, to barely noticeable effect:

```{code-cell} ipython3
X = ads[["Radio","TV"]]
lm.fit(X,y)
print("R^2 score:",f"{lm.score(X,y):.4f}")
print("Model coeffs:",lm.coef_)
print(pd.Series(lm.coef_,index=X.columns))
```

While we have a good $R^2$ score, there is some unexplained variance remaining. We can add an additional feature that is the product of *TV* and *Radio*, representing the possibility that these media reinforce one another's effects. (In order to modify our frame `X`, which is only a view of part of the original frame `ads`, we first have to redefine it as an independent copy.)

```{code-cell} ipython3
X = ads[["Radio","TV"]].copy()
X["RadioTV"] = X["Radio"]*X["TV"]
lm.fit(X,y)
print("R^2 score:",f"{lm.score(X,y):.4f}")
print("Model coeffs:")
print(pd.Series(lm.coef_,index=X.columns))
```

We did see some increase in the $R^2$ score, and therefore the combination of both types of spending does have a positive effect on *Sales*. We have to be careful interpreting the magnitudes of the coefficients, because the size of the product feature is 100 or so times greater than either individual constituent. In that light, the interaction effect seems comparable to the individual features. 

Interpreting linear regression is a major topic in statistics. There are tests that can lend more precision and rigor to the brief discussion above.

## Polynomial regression

An important special case of multilinear regression is when there is initially a single predictor variable $t$, and then we define

$$
x_1 = t^0, \, x_2 = t^1, \ldots, x_d = t^{d-1}.
$$

This makes the regressive approximation into

$$
y \approx w_1 + w_2 t + \cdots + w_d t^{d-1},
$$

which is a polynomial of degree $d-1$. This allows representation of data that depends on $t$ in ways more complicated than a straight line. But it can also introduce variance.

## Case study: Fuel efficiency

We return to the data set regarding the fuel efficiency of cars.

```{code-cell} ipython3
import seaborn as sns
cars = sns.load_dataset("mpg").dropna()
cars
```

As we would expect, horsepower and miles per gallon are negatively correlated. However, the relationship is not well captured by a straight line.

```{code-cell} ipython3
sns.lmplot(data=cars,x="horsepower",y="mpg");
```

A cubic polynomial produces a much more plausible fit, especially on the right half of the plot.

```{code-cell} ipython3
sns.lmplot(data=cars,x="horsepower",y="mpg",order=3);
```

In order to produce the cubic fit in sklearn, we use the `PolynomialFeatures` preprocessor in a pipeline. If the original horsepower predictor variable is $t$, then the preprocessor will create features for $1$, $t$, $t^2$, and $t^3$. (Since the constant feature is added in, we don't need to fit the intercept with the linear regressor.)

```{code-cell} ipython3
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import pandas as pd

X = cars[["horsepower"]]
y = cars["mpg"]
lm = LinearRegression(fit_intercept=False)
cubic = make_pipeline(PolynomialFeatures(degree=3),lm)
cubic.fit(X,y)

query = pd.DataFrame([200],columns=X.columns)
print("prediction at hp=200:",cubic.predict(query))
```

The prediction above is consistent with the earlier figure. 

We can get the coefficients of the cubic polynomial from the trained regressor:

```{code-cell} ipython3
cubic[1].coef_
```

The coefficients go in order of increasing degree.

If a cubic polynomial can fit better than a line, it's plausible that increasing the degree more will lead to even better fits. In fact, the training error can only go down, because a lower-degree polynomial case is a subset of a higher-degree case.

To explore the effect of degree, we split into train and test sets.

```{code-cell} ipython3
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X_tr,X_te,y_tr,y_te = train_test_split(X,y,test_size=0.2,random_state=0)

for deg in range(2,11):
    poly = make_pipeline(PolynomialFeatures(degree=deg),lm)
    poly.fit(X_tr,y_tr)
    print(f"MSE for degree {deg}:",mean_squared_error(y_te,poly.predict(X_te)))
```

The results above are a classic example of overfitting and the bias–variance tradeoff. A plot of the degree-10 fit shows that the polynomial becomes more oscillatory:

```{code-cell} ipython3
sns.lmplot(data=cars,x="horsepower",y="mpg",order=10);
```

In the above plot, note the widening of the confidence intervals near the ends of the domain, indicating increased variance in the predictions. 

Next, we keep more of the original data features and pursue a multilinear fit. We chain it with a `StandardScaler` so that all columns have equal mean and scale.

```{code-cell} ipython3
from sklearn.preprocessing import StandardScaler

X = cars[["horsepower","displacement","cylinders","weight"]]
y = cars["mpg"]
X_tr,X_te,y_tr,y_te = train_test_split(X,y,test_size=0.2,random_state=0)

lm = LinearRegression(fit_intercept=True)
pipe = make_pipeline(StandardScaler(),lm)
pipe.fit(X_tr,y_tr)
print(f"MSE for multilinear:",mean_squared_error(y_te,pipe.predict(X_te)))
```

The fit here is actually a little worse than the low-degree fits based on horsepower alone. However, by comparing the coefficients of the individual features, some interesting information emerges:

```{code-cell} ipython3
pd.Series(pipe[1].coef_,index=X.columns)
```

We now have a hypothesis that weight is the most significant negative factor for MPG, and by a wide margin.

Finally, we can combine the use of multiple features and higher degree.

```{code-cell} ipython3
pipe = make_pipeline(StandardScaler(),PolynomialFeatures(degree=2),lm)
pipe.fit(X_tr,y_tr)
print(f"MSE for multilinear:",mean_squared_error(y_te,pipe.predict(X_te)))
```

This is our best regression fit so far.

<div style="max-width:608px"><div style="position:relative;padding-bottom:66.118421052632%"><iframe id="kaltura_player" src="https://cdnapisec.kaltura.com/p/2358381/sp/235838100/embedIframeJs/uiconf_id/43030021/partner_id/2358381?iframeembed=true&playerId=kaltura_player&entry_id=1_wg4yj0hs&flashvars[streamerType]=auto&amp;flashvars[localizationCode]=en&amp;flashvars[leadWithHTML5]=true&amp;flashvars[sideBarContainer.plugin]=true&amp;flashvars[sideBarContainer.position]=left&amp;flashvars[sideBarContainer.clickToClose]=true&amp;flashvars[chapters.plugin]=true&amp;flashvars[chapters.layout]=vertical&amp;flashvars[chapters.thumbnailRotator]=false&amp;flashvars[streamSelector.plugin]=true&amp;flashvars[EmbedPlayer.SpinnerTarget]=videoHolder&amp;flashvars[dualScreen.plugin]=true&amp;flashvars[Kaltura.addCrossoriginToIframe]=true&amp;&wid=1_68o0vx38" width="608" height="402" allowfullscreen webkitallowfullscreen mozAllowFullScreen allow="autoplay *; fullscreen *; encrypted-media *" sandbox="allow-forms allow-same-origin allow-scripts allow-top-navigation allow-pointer-lock allow-popups allow-modals allow-orientation-lock allow-popups-to-escape-sandbox allow-presentation allow-top-navigation-by-user-activation" frameborder="0" title="Kaltura Player" style="position:absolute;top:0;left:0;width:100%;height:100%"></iframe></div></div>
