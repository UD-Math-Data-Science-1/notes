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

```{code-cell} ipython3
import seaborn as sns
cars = sns.load_dataset("mpg")
cars = cars.dropna()
cars
```

```{code-cell} ipython3
sns.relplot(data=cars,x="horsepower",y="mpg")
```

```{code-cell} ipython3
sns.lmplot(data=cars,x="horsepower",y="mpg")
```

```{code-cell} ipython3
sns.lmplot(data=cars,x="horsepower",y="mpg",order=3)
```

```{code-cell} ipython3
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import numpy as np

X = np.array(cars["horsepower"]).reshape(-1,1)
y = cars["mpg"]
lm = LinearRegression(fit_intercept=False)
cubic = make_pipeline(PolynomialFeatures(degree=3),lm)
cubic.fit(X,y)

print("prediction at hp=100:",cubic.predict([[200]]))
```

```{code-cell} ipython3
cubic[1].coef_
```

```{code-cell} ipython3
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X_tr,X_te,y_tr,y_te = train_test_split(X,y,test_size=0.2,random_state=0)
```

```{code-cell} ipython3
from sklearn.preprocessing import StandardScaler

for deg in range(2,11):
    poly = make_pipeline(PolynomialFeatures(degree=deg),lm)
    poly.fit(X_tr,y_tr)
    print(f"MSE for degree {deg}:",mean_squared_error(y_te,poly.predict(X_te)))
```

```{code-cell} ipython3
sns.lmplot(data=cars,x="horsepower",y="mpg",order=10)
```

```{code-cell} ipython3
poly[1].coef_
```

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

```{code-cell} ipython3
print(pipe[1].coef_)
print(X.columns)
```

```{code-cell} ipython3
pipe = make_pipeline(StandardScaler(),PolynomialFeatures(degree=2),lm)
pipe.fit(X_tr,y_tr)
print(f"MSE for multilinear:",mean_squared_error(y_te,pipe.predict(X_te)))
```

```{code-cell} ipython3

```
