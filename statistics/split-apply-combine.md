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

# Split–apply–combine

One of the most important workflows in data analysis is called **split–apply–combine**:

1. Split the data into groups based on a criterion (e.g., species, marital status).
2. Apply operations to the data within each group.
3. Combine the results from the groups.

Of these, the *apply* step is usually the most complex.

+++

## Split

The `groupby` method for a data frame splits the frame into groups based on categorical values in a designated column. For illustrations, we will load a dataset supplied with seaborn. 

```{code-cell} ipython3
import pandas as pd
import seaborn as sns

penguins = sns.load_dataset("penguins")
penguins
```

```{code-cell} ipython3
by_species = penguins.groupby("species")
```

Nothing is actually done yet to the data frame. It's just set up for applying operations to each group.

```{code-cell} ipython3
for name,group in by_species:
    print(name)
    print(group.iloc[:3,:4])
    print()
```

A common operation is to "bin" the values of a continuous variable into intervals. You can group by bins using `cut`.

```{code-cell} ipython3
cuts = pd.cut(penguins["bill_length_mm"],[0,30,40,50,100])
by_length = penguins.groupby(cuts)
for name,group in by_length:
    print(name)
    print(group.iloc[:3,:4])
    print()
```

## Apply

The most complex step is applying operations to each group of data. There are three types of operations:

* **Aggregation** refers to summarizing data by a single value, such as a sum or mean, or by a few values, such as value counts or quintiles.
* **Transformation** refers to application of a mathematical operation to every data value, resulting in data indexed the same way as the original. For example, quantitative data might be transformed to lie in the interval $[0,1]$.
* **Filtration** refers to inclusion/removal of a group based on a criterion, such as rejection of a group with too few members.

+++

### Aggregation

Many common operations are defined for aggregation.

```{code-cell} ipython3
by_length.sum()
```

```{code-cell} ipython3
by_length.count()
```

Note that after grouping has been done, you can refer to subsets of values within the group in the usual way.

```{code-cell} ipython3
by_species["island"].value_counts()
```

Statistical summaries can also be computed by group.

```{code-cell} ipython3
by_species.mean()
```

```{code-cell} ipython3
by_species["body_mass_g"].describe()
```

A list of the most common predefined aggregation functions is given in {numref}`table-aggregators`. These functions ignore `NaN` (missing) values. 

```{list-table} Aggregation functions
:name: table-aggregators
* - `mean`
  - Mean of group values
* - `sum`
  - Sum of group values
* - `count`
  - Count of group values
* - `std`, `var`
  - Standard deviation or variance within groups
* - `describe`
  - Descriptive statistics
* - `first`, `last`
  - First or last of group values
* - `min`, `max`
  - Min or max within groups
```

+++

If you want a more exotic operation, you can call `agg` with your own function.

```{code-cell} ipython3
def iqr(x):
    q1,q3 = x.quantile([.25,.75])
    return q3-q1

by_length["bill_length_mm"].agg(iqr)
```

### Transformation

In the simplest case, a transformation applies a function to each element of a column, producing a result of the same length that can be indexed the same way. For example, we can standardize to z-scores within each group separately, rather than based on the global mean and std. 

```{code-cell} ipython3
def standardize(x):
    return (x-x.mean())/x.std()

penguins["group_z"] = by_species["bill_length_mm"].transform(standardize)
sns.displot(data=penguins,x="group_z",col="species");
```

If we standardize instead by the global statistics, then the group distributions won't be standardized.

```{code-cell} ipython3
penguins["global_z"] = standardize(penguins["bill_length_mm"])
sns.displot(data=penguins,x="global_z",col="species");
```

### Filtering

To apply a filter, provide a function that operates on a column and returns either `True`, meaning to keep the column, or `False`, meaning to reject it. For example, suppose we want to group penguins by body mass. 

```{code-cell} ipython3
cuts = pd.cut(penguins["body_mass_g"],range(2000,7000,500))
by_mass = penguins.groupby(cuts)
by_mass["species"].count()
```

If we want to drop the penguins that fall into groups having fewer than 30 penguins, we use `filter`.

```{code-cell} ipython3
mass_30 = by_mass.filter(lambda x: len(x) > 29)
mass_30
```

Notice that the result has been merged (ungrouped) back into a single frame. We can regroup the result, however.

```{code-cell} ipython3
cuts = pd.cut(mass_30["body_mass_g"],range(2000,7000,500))
mass_30.groupby(cuts)["species"].count()
```

## Combine

Pandas assembles the results of an application into a series or data frame, depending on the context. Sometimes, you might want to store those values in the original data frame, as shown above with group standardization.

It's common to form chains of operations that can be written separately or in one line. For example, consider the task of **imputation**, which is the replacement of missing values by a standard value such as the mean or median. In the penguin dataset, there are two rows with missing numerical values:

```{code-cell} ipython3
bills = ["bill_length_mm","bill_depth_mm"]
penguins[bills].isna().sum()
```

Given variations between species, we probably want to compute values aggregated by species. 

```{code-cell} ipython3
by_species = penguins.groupby("species")
by_species[bills].median()
```

In order to operate columnwise, we apply a custom transformation function using the `fillna` method to replace missing values. 

```{code-cell} ipython3
def impute(col):
    return col.fillna(col.median())

replaced = by_species[bills].transform(impute)
replaced
```

Replacement has happened in the row with index 3, for example. Finally, we can overwrite the columns of the original data frame, if we don't care to know in the future which values were imputed. All of the necessary steps can be compressed into one chain:

```{code-cell} ipython3
penguins[bills] = penguins.groupby("species")[bills].transform(lambda x: x.fillna(x.median()))
penguins
```
