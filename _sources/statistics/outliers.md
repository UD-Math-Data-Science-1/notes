---
jupytext:
  formats: md:myst
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

# Outliers

Informally, an **outlier** is a data value that is considered to be far from typical. In some applications, such as detecting earthquakes or cancer, outliers are the cases of real interest. But we will be thinking of them as unwelcome values that might result from equipment failure, confounding effects, mistyping a value, using an extreme value to represent missing data, and so on. In such cases we want to minimize the effect of the outliers on the statistics. 

There are various ways of deciding what "typical" means, and there is no one-size recommendation for all applications. 

## IQR

Let's look at another data set, based on an fMRI experiment.

```{code-cell}
import pandas as pd
import seaborn as sns

fmri = sns.load_dataset("fmri")
fmri.head()
```

We want to focus on the *signal* column, splitting according to the *event*.

```{code-cell}
fmri.groupby("event")["signal"].describe()
```

Here is a box plot of the signal for these groups.

```{code-cell}
sns.catplot(data=fmri,x="event",y="signal",kind="box")
```

The dots lying outside the whiskers in the plot may be considered outliers. They are determined by the quartiles. Let $Q_1$ and $Q_3$ be the first and third quartiles (i.e., 25% and 75% percentiles), and let $I=Q_3-Q_1$ be the interquartile range (IQR). Then $x$ is an outlier value if

$$ 
x < Q_1 - 1.5I \text{ or } x > Q_3 + 1.5I.
$$


## Mean and STD

For normal distributions, values more than twice the standard deviation $\sigma$ from the mean might be declared to be outliers; this would exclude 5% of the values, on average. A less aggressive criterion is to allow a distance of $3\sigma$, which excludes only about 0.3% of the values. The IQR criterion above corresponds to about $2.7\sigma$ in the normal case.

The following plot shows the outlier cutoffs for 2000 samples from a normal distribution, using the criteria for 2σ (red), 3σ (blue), and 1.5 IQR (black).

```{code-cell}
:tags: [hide-input]

import matplotlib.pyplot as plt
from numpy.random import default_rng
randn = default_rng(1).normal 

x = pd.Series(randn(size=2000))
sns.displot(data=x,bins=30);
m,s = x.mean(),x.std()
plt.axvline(m-2*s,color='r')
plt.axvline(m+2*s,color='r')
plt.axvline(m-3*s,color='b')
plt.axvline(m+3*s,color='b')

q1,q3 = x.quantile([.25,.75])
plt.axvline(q3+1.5*(q3-q1),color='k')
plt.axvline(q1-1.5*(q3-q1),color='k');
```

For asymmetric distributions, or those with a fat tail, these criteria might show greater differences.

## Removing outliers

It is well known that the mean is more sensitive to outliers than the median is. 

```{prf:example}
The values $1,2,3,4,5$ have a mean and median both equal to 3. If we change the largest value to be a lot larger, say $1,2,3,4,1000$, then the mean changes to 202. But the median is still 3!
```

Let's use IQR to remove outliers from the fmri data set. We do this by creating a Boolean-valued series indicating which rows of the frame represent outliers within their group.

```{code-cell}
def isoutlier(x):
    Q1,Q3 = x.quantile([.25,.75])
    I = Q3-Q1
    return (x < Q1-1.5*I) |  (x > Q3+1.5*I)

outs = fmri.groupby("event")["signal"].transform(isoutlier)
fmri[outs]["event"].value_counts()
```

You can see above that there are 66 outliers. To negate the outlier indicator, we can use `~outs` as a row selector.

```{code-cell}
cleaned = fmri[~outs]
```

The median values are barely affected by the omission of the outliers.

```{code-cell}
print("medians with outliers:")
print(fmri.groupby("event")["signal"].median())
print("\nmedians without outliers:")
print(cleaned.groupby("event")["signal"].median())
```

The means show much greater change.

```{code-cell}
print("means with outliers:")
print(fmri.groupby("event")["signal"].mean())
print("\nmeans without outliers:")
print(cleaned.groupby("event")["signal"].mean())
```

For the *stim* case in particular, the mean value changes by almost 200%, including a sign change. (Relative to the standard deviation, it's closer to a 20% change.)  

<div style="max-width:608px"><div style="position:relative;padding-bottom:66.118421052632%"><iframe id="kaltura_player" src="https://cdnapisec.kaltura.com/p/2358381/sp/235838100/embedIframeJs/uiconf_id/43030021/partner_id/2358381?iframeembed=true&playerId=kaltura_player&entry_id=1_vpsvig7f&flashvars[streamerType]=auto&amp;flashvars[localizationCode]=en&amp;flashvars[leadWithHTML5]=true&amp;flashvars[sideBarContainer.plugin]=true&amp;flashvars[sideBarContainer.position]=left&amp;flashvars[sideBarContainer.clickToClose]=true&amp;flashvars[chapters.plugin]=true&amp;flashvars[chapters.layout]=vertical&amp;flashvars[chapters.thumbnailRotator]=false&amp;flashvars[streamSelector.plugin]=true&amp;flashvars[EmbedPlayer.SpinnerTarget]=videoHolder&amp;flashvars[dualScreen.plugin]=true&amp;flashvars[Kaltura.addCrossoriginToIframe]=true&amp;&wid=1_a1t3d8un" width="608" height="402" allowfullscreen webkitallowfullscreen mozAllowFullScreen allow="autoplay *; fullscreen *; encrypted-media *" sandbox="allow-forms allow-same-origin allow-scripts allow-top-navigation allow-pointer-lock allow-popups allow-modals allow-orientation-lock allow-popups-to-escape-sandbox allow-presentation allow-top-navigation-by-user-activation" frameborder="0" title="Kaltura Player" style="position:absolute;top:0;left:0;width:100%;height:100%"></iframe></div></div>
