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

# Using scikit-learn

The scikit-learn package (`sklearn`) is a collection of machine learning algorithms and tools. It includes a few classic example datasets. We will load one derived from automatic recognition of handwritten digits. 

```{code-cell} ipython3
from sklearn import datasets 
ds = datasets.load_digits()        # loads a well-known dataset
X,y = ds["data"],ds["target"]      # assign feature matrix and label vector
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
knn = KNeighborsClassifier(n_neighbors=20)   # specification of the model
knn.fit(X,y)                                 # training of the model
```

At this point, the classifier object `knn` has figured out what it needs to do with the training data. It has methods we can call to make predictions and evaluate the quality of the results. Each new prediction is for a *query vector* with 64 components (features). The `predict` method of the classifier allows specifying multiple query vectors as rows of an array—in fact, it expects a 2D array in all cases, even if there is just one row.

```{code-cell}
query = [20]*d   # list, d copies of 20
Xq = [ query ]   # 2D array with a single row
# Get vector of predictions:
knn.predict(Xq)  
```

We don't have any realistic query data at hand other than the training data. But by comparing the predictions made for that data to the true labels we supplied, we can get some idea of how accurate the predictor is.

```{code-cell}
# Get vector of predictions on the original set:
yhat = knn.predict(X)    
yhat[-6:]      # last six components
```

Compared to the true labels we printed out above, so far, so good. Now we simply count up the number of correctly predicted labels and divide by the total number of samples. 

```{code-cell}
acc = sum(yhat==y)/n    # fraction of correct predictions
print(f"accuracy is {acc:.1%}")
```

Not surprisingly, sklearn has functions for doing this measurement in fewer steps. The `metrics` module has functions that can compare true labels with predictions. In addition, each classifier object has a `score` method that allows you to skip finding the predictions vector yourself.

```{code-cell}
from sklearn.metrics import accuracy_score

# Compare original labels to predictions:
acc = accuracy_score(y,yhat)    
print(f"accuracy score is {acc:.1%}")

# Compute accuracy on the original dataset (same result):
acc = knn.score(X,y)    
print(f"knn score is {acc:.1%}")
```

## Train–test paradigm

Good performance of a classifier on the samples used to train seems to be necessary, but is it sufficient? We are more interested on how the classifier performs on new data. This is the question of *generalization*. In order to gauge generalization, we hold back some of the labeled data from training and use it only to test the performance.


A `sklearn` helper function allows us to split off a random 20% of the data to use for testing. By default, it will preserve the order of the test set. This can be a problem if, for example, the samples are presented already sorted by class. It's usually recommended to shuffle the order first, but here we give a specific random seed so that the results are reproducible.

```{code-cell}
from sklearn.model_selection import train_test_split

X_tr, X_te, y_tr, y_te = train_test_split(X,y,
  test_size=0.2,
  shuffle=True,random_state=0)
```

We can check that the test and train labels have similar characteristics:

```{code-cell}
import pandas as pd
print("training:")
print(pd.Series(y_tr).describe())

print("\ntest:")
print(pd.Series(y_te).describe())
```

Now we train on the training data...

```{code-cell}
knn = KNeighborsClassifier(n_neighbors=20)
knn.fit(X_tr,y_tr)   # fit only to train set
```

...and test on the rest.

```{code-cell}
acc = knn.score(X_te,y_te)   # score only on test set
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
loans.isnull().sum()
```


Finally, we'll output this cleaned data frame to its own CSV file. The index row is an ID number that is meaningless to classification, so we will exclude it from the new file.
```{code-cell}
loans.to_csv("loan_clean.csv",index=False)
```

<div style="max-width:608px"><div style="position:relative;padding-bottom:66.118421052632%"><iframe id="kaltura_player" src="https://cdnapisec.kaltura.com/p/2358381/sp/235838100/embedIframeJs/uiconf_id/43030021/partner_id/2358381?iframeembed=true&playerId=kaltura_player&entry_id=1_o5cvngqc&flashvars[streamerType]=auto&amp;flashvars[localizationCode]=en&amp;flashvars[leadWithHTML5]=true&amp;flashvars[sideBarContainer.plugin]=true&amp;flashvars[sideBarContainer.position]=left&amp;flashvars[sideBarContainer.clickToClose]=true&amp;flashvars[chapters.plugin]=true&amp;flashvars[chapters.layout]=vertical&amp;flashvars[chapters.thumbnailRotator]=false&amp;flashvars[streamSelector.plugin]=true&amp;flashvars[EmbedPlayer.SpinnerTarget]=videoHolder&amp;flashvars[dualScreen.plugin]=true&amp;flashvars[Kaltura.addCrossoriginToIframe]=true&amp;&wid=1_vt4ow2xr" width="608" height="402" allowfullscreen webkitallowfullscreen mozAllowFullScreen allow="autoplay *; fullscreen *; encrypted-media *" sandbox="allow-forms allow-same-origin allow-scripts allow-top-navigation allow-pointer-lock allow-popups allow-modals allow-orientation-lock allow-popups-to-escape-sandbox allow-presentation allow-top-navigation-by-user-activation" frameborder="0" title="Kaltura Player" style="position:absolute;top:0;left:0;width:100%;height:100%"></iframe></div></div>