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

The scikit-learn package (`sklearn`) is a collection of machine learning algorithms and tools. It includes a few classic example datasets. We will load one derived from automatic recognition of handwritten digits. 

```{code-cell} ipython3
from sklearn import datasets 
ds = datasets.load_digits()
X,y = ds["data"],ds["target"]
print("feature matrix has shape",X.shape)
print("label vector has shape",y.shape)
n,d = X.shape
print("there are",d,"features and",n,"samples")
```

Here, we look at the first 5 features of the first instance.

```{code-cell}
X[0,:5]
```

And here are the last 6 labels.

```{code-cell}
y[-6:]
```

There are three main activities supported by sklearn:

* **fit**, to train the classifier
* **predict**, to apply the classifier
* **transform**, to modify the data

We'll explore fitting and prediction for now. Let's try a classifier whose characteristics we will explain in a future section.

```{code-cell}
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=20)   # specification
knn.fit(X,y)                                 # training
```

At this point, the classifier object `knn` has figured out what it needs to do with the training data, and we can ask it to make predictions. Each application we want a prediction for is a vector with 64 components (features). The prediction query has to be a 2D array, with each row being a query vector. The result is a vector of predictions.

```{code-cell}
Xq = [ [20]*d ]
knn.predict(Xq)
```

We don't have any realistic data at hand other than the training data. By comparing the predictions made for that data to the true labels we supplied, we can get some idea of how accurate the predictor is.

```{code-cell}
yhat = knn.predict(X)   # prediction
yhat[-6:]
```

Compared to the true labels we printed out above, so far, so good. Now we simply count up the number of correctly predicted labels and divide by the total number of samples. 

```{code-cell}
acc = sum(yhat==y)/n 
print(f"accuracy is {acc:.1%}")
```

Of course, sklearn has functions for doing this measurement in fewer steps.

```{code-cell}
from sklearn import metrics

acc = metrics.accuracy_score(y,yhat)
print(f"accuracy score is {acc:.1%}")

acc = knn.score(X,y)
print(f"knn score is {acc:.1%}")
```

## Train–test paradigm

Good performance of a classifier on the samples used to train seems to be necessary, but is it sufficient? We are more interested on how the classifier performs on new data. This is the question of *generalization*. In order to gauge generalization, we hold back some of the labeled data from training and use it only to test the performance.


A `sklearn` helper function allows us to split off a randomized 20% of the data to use for testing:

```{code-cell}
from sklearn.model_selection import train_test_split
X_tr, X_te, y_tr, y_te = train_test_split(X,y,test_size=0.2,shuffle=True,random_state=0)
print(len(y_tr),"training cases and",len(y_te),"test cases")
```

Now we train on the training data...

```{code-cell}
from sklearn import neighbors as nbr
knn = nbr.KNeighborsClassifier(n_neighbors=20)
knn.fit(X_tr,y_tr)
```

...and test on the rest.

```{code-cell}
acc = knn.score(X_te,y_te)
print(f"accuracy is {acc:.1%}")
```

This is fairly consistent with the accuracy we found before by training and testing on the full sample set. As we will see later, that happy situation does not always hold.

## sklearn and pandas

Scikit-learn plays very nicely with pandas. You can use data frames for the sample values and labels. Let's look at a data set that comes from seaborn.

```{code-cell}
import seaborn as sns
penguins = sns.load_dataset("penguins")
penguins = penguins.dropna()
penguins
```

The `dropna` call above removes rows that have a `NaN` value in any column, as many sklearn learning algorithms aren't able to handle them. This data frame has four quantitative columns that we will use as features. We will use the species column for the labels. 

```{code-cell}
X = penguins[["bill_length_mm","bill_depth_mm","flipper_length_mm","body_mass_g"]]
y = penguins["species"]
```

Now `X` is a data frame and `y` is a series. They can be input directly into a learning method call.

```{code-cell}
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X,y);
```

To make predictions, we need to pass in another data frame that has the same columns as `X`. Here is a simple way to turn a numerical vector or list into such a frame.

```{code-cell}
import pandas as pd
x_new = [39,19,180,3750]
xdf = pd.DataFrame([x_new],columns=X.columns)
xdf
```

(The `[x_new]` part does need to have the brackets, so that pandas sees a list of row values there. We could put multiple rows in that list.) Now we can use the classifier to make a prediction.

```{code-cell}
knn.predict(xdf)
```

The result comes back as a series of the same dtype as `y`.

This may seem like hassle and extra work for mere window dressing. Why not just work with arrays? When we are focusing on the math, that's fine. But in an application, it's easy to lose track of what the integer indexes of an array are supposed to mean. By using their names, you keep things clearer in your code and own mind, and sklearn will give you warnings and errors if you aren't using them consistently. In other words, it's *productive* hassle.

## Data cleaning

Raw data often needs to be manipulated into a useable numerical format before algorithms can be applied. We will go through some of these steps for a dataset describing loans made on the crowdfunding site LendingClub.

First, we load the raw data from a CSV (comma separated values) file. 

```{code-cell}
import pandas as pd
loans = pd.read_csv("loan.csv")
loans.head()
```

The `int_rate` column, which gives the interest rate on the loan, has been interpreted as strings due to the percent sign. We'll strip out those percent signs, and then sklearn will handle the conversion to numbers as needed.

```{code-cell}
loans["rate"] = loans["int_rate"].str.strip('%')
```

Let's add a column for the percentage of the loan request that was eventually funded. This will be a target for some of our learning methods.

```{code-cell}
loans["percent_funded"] = 100*loans["funded_amnt"]/loans["loan_amnt"]
target = ["percent_funded"]
```

We will only use a small subset of the numerical columns as features. Let's verify that there are no missing values in those columns.

```{code-cell}
features = [ "loan_amnt","rate","installment","annual_inc","dti","delinq_2yrs","delinq_amnt"]
loans = loans.loc[:,features+target]
loans.isna().sum()
```


Finally, we'll output this cleaned data frame to its own CSV file.
```{code-cell}
loans.to_csv("loan_clean.csv")
```
