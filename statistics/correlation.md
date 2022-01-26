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
# Correlation

For illustrations, let us load a data set about penguins.

```{code-cell} ipython3
import pandas as pd
import seaborn as sns
penguins = sns.load_dataset("penguins")
penguins
```

There are often observations that we believe to be linked, either because one influences the other, or both are influenced by some other factor. That is, we say the quantities are **correlated**. This can become apparent using `pairplot` in seaborn.

```{code-cell}
sns.pairplot(penguins,hue="species");
```

The panels along the diagonal show each quantitative variable's distribution as a KDE plot. The other panels show scatter plots putting one pair at a time of the variables on the coordinate axes. There appears to be a strong positive correlation between flipper length and body mass in all three species, while the relationship between flipper length and bill length is less clear.

There are several ways to measure correlation.

## Covariance

Suppose we have two series of observations, $[x_i]$ and $[y_i]$, representing observations of random quantities $X$ and $Y$ having means $\mu_X$ and $\mu_Y$. Then the values $[x_i-\mu_X]$ and $[y_i-\mu_Y]$ are deviations from the means. The **covariance** of the quantities is defined as 

$$
\Cov(X,Y) = \frac{1}{n} \sum_{i=1}^n (x_i-\mu_X)(y_i-\mu_Y).
$$

One explanation for the name is that $\Cov(X,X)$ and $\Cov(Y,Y)$ are just the variances of $X$ and $Y$. 

Covariance is not easy to interpret. Its units are the products of the units of the two variables, and it is sensitive to rescaling the variables (e.g., grams versus kilograms).

## Pearson correlation coefficient

We can remove the dependence on units and scale by applying the covariance to standardized scores for both variables:

$$
\rho(X,Y) = \frac{1}{n} \sum_{i=1}^n \left(\frac{x_i-\mu_X}{\sigma_X}\right)\left(\frac{y_i-\mu_Y}{\sigma_Y}\right)
= \frac{\Cov(X,Y)}{\sigma_X\sigma_Y},
$$

where $\sigma_X^2$ and $\sigma_Y^2$ are the variances of $X$ and $Y$. The value of $\rho$, called the **Pearson correlation coefficient**, is between $-1$ and $1$, with the endpoints indicating perfect correlation (negative or positive). 

For example, we might reasonably expect flipper length and body mass to be correlated in penguins, as a plot confirms:

```{code-cell}
sns.relplot(data=penguins,x="flipper_length_mm",y="body_mass_g");
```

Covariance allows us to confirm a positive relationship:

```{code-cell}
flip = penguins["flipper_length_mm"]
mass = penguins["body_mass_g"]

flip.cov(mass)
```

But is that a lot? The Pearson coefficient is more helpful.

```{code-cell}
flip.corr(mass)
```

The value of about $0.87$ suggests that knowing one of the values would allow us to predict the other one rather well using a best-fit straight line (more on that in a future chapter).

As usual when dealing with means, however, the Pearson coefficient can be sensitive to outlier values. For example, let's correlate two series that differ in only one element: $0,1,2,\ldots,19$, and the same sequence with the fifth value replaced by $-100$.

```{code-cell}
x = pd.Series(range(20))
y = x.copy()
y[4] = -100
x.corr(y)
```

Over half of the predictive value was lost. 

## Spearman coefficient

The Spearman coefficient is one way to lessen the impact of outliers when measuring correlation. The idea is that the values are used only in their relationship to one another. 

Define $r_i$ as the position of $x_i$ in a sorted reordering of the sampled values of $X$. Similarly, let $s_i$ be a position or *rank* series for the values of $Y$. Then the **Spearman coefficient** is defined as the Pearson coefficient of the variables $R$ and $S$.

For the example above, it's trivial to produce the rank series by hand.

```{code-cell}
r = pd.Series(range(1,21))
s = r.copy()
s[:5] = [2,3,4,5,1]

r.corr(s)
```

This value is still very close to perfect correlation. pandas has a method for doing this calculation automatically on the original series.

```{code-cell}
x.corr(y,"spearman")
```

As long as `y[4]` is negative, it doesn't matter what its particular value is, because that has no effect on the ranking.

```{code-cell}
y[4] = -1000000
x.corr(y,"spearman")
```

Since real data almost always features outlying or anomalous values, it's important to think about the robustness of the statistics you choose.

## Categorical correlation

An ordinal variable, such as days of the week, are often straightforward to quantify as consecutive integers. But a nominal variable poses a different challenge. 

For example, grouped histograms suggest an association between body mass and species of penguin.

```{code-cell}
sns.displot(data=penguins,x="body_mass_g",col="species");
```

How can we quantify the association? The first step is to convert the species column into dummy variables.

```{code-cell}
dum = pd.get_dummies(penguins,columns=["species"])
dum
```

The original species column has been replaced by three binary indicator columns. Now we can look for correlations between them and the body mass:

```{code-cell}
dum[["body_mass_g","species_Adelie","species_Chinstrap","species_Gentoo"]].corr()
```

As you can see from the above, Adelie and (to a lesser extent) Chinstrap are associated with lower mass, while Gentoo is strongly associated with higher mass.

## Simpson's paradox

We can find all the pairwise correlation coefficients in the same style as the grid of pair plots at the top of this section.

```{code-cell}
penguins.corr()
```

For example, the correlation between body mass and bill depth is about $-0.472$. But something interesting happens if we compute the correlations *after* grouping by species.

```{code-cell}
penguins.groupby("species").corr()
```

Within each individual species, the correlation between body mass and bill depth is greater than $0.5$!
This is an example of **Simpson's paradox**. The reason for it can be seen from the pair plot above. Within each color, there is a strong positive association. But the relationship isn't identical across species, and what dominates the combination of all three is the large gap between the Gentoo and the other species.

Simpson's paradox shows how important it us to understand the dataset before spewing out statistics about it. There are contexts where combining species of penguins makes sense, but in the case of body mass, we are really dealing with three separate distributions.

