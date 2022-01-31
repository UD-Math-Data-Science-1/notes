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

# Introduction to pandas

The most popular Python package for manipulating and analyzing data is [pandas](https:pandas.pydata.org). In particular, it centers on the series–data frame paradigm. 

For example, here is a series for the wavelengths of light corresponding to rainbow colors.

```{code-cell} ipython3
import pandas as pd
wave = pd.Series([400,470,520,580,610,710],index=["violet","blue","green","yellow","orange","red"])
print(wave)
```

We can now use an index value to access a value in the series.

```{code-cell} ipython3
print(wave["blue"])
```

We can access multiple values to get a series that is a subset of the original.

```{code-cell} ipython3
print(wave[["violet","red"]])
```

Here is a series of NFL teams based on the same index.

```{code-cell} ipython3
teams = pd.Series(["Vikings","Bills","Eagles","Chargers","Bengals","Cardinals"],index=wave.index)
print(teams["green"])
```

Now we can create a data frame using these two series as columns.

```{code-cell} ipython3
rainbow = pd.DataFrame({"wavelength":wave,"team name":teams})
print(rainbow)
```

We can add a column after the fact just by giving the values. The indexing is inherited from the current frame.

```{code-cell} ipython3
rainbow["flower"] = ["Lobelia","Cornflower","Bells-of-Ireland","Daffodil","Butterfly weed","Rose"]
print(rainbow)
```

Interestingly, a row of the data frame (accessed using `loc` below) is itself a series, indexed by the column names.

```{code-cell} ipython3
print(rainbow.loc["orange"])
```

There are many ways to specify values for a data frame. If no explicit index set is given, then consecutive integers starting at zero are used.

```{code-cell} ipython3
letters = pd.DataFrame([("a","A"),("b","B"),("c","C")],columns=["lowercase","uppercase"])
print(letters)
```

```{code-cell} ipython3
print(letters.loc[1])
```

```{code-cell} ipython3
print(letters["uppercase"])
```

For much more about pandas fundamentals, try the [Kaggle course](https://www.kaggle.com/learn/pandas).

<div style="max-width:608px"><div style="position:relative;padding-bottom:66.118421052632%"><iframe id="kaltura_player" src="https://cdnapisec.kaltura.com/p/2358381/sp/235838100/embedIframeJs/uiconf_id/43030021/partner_id/2358381?iframeembed=true&playerId=kaltura_player&entry_id=1_z2y5ubh4&flashvars[streamerType]=auto&amp;flashvars[localizationCode]=en&amp;flashvars[leadWithHTML5]=true&amp;flashvars[sideBarContainer.plugin]=true&amp;flashvars[sideBarContainer.position]=left&amp;flashvars[sideBarContainer.clickToClose]=true&amp;flashvars[chapters.plugin]=true&amp;flashvars[chapters.layout]=vertical&amp;flashvars[chapters.thumbnailRotator]=false&amp;flashvars[streamSelector.plugin]=true&amp;flashvars[EmbedPlayer.SpinnerTarget]=videoHolder&amp;flashvars[dualScreen.plugin]=true&amp;flashvars[Kaltura.addCrossoriginToIframe]=true&amp;&wid=1_p2kbu4tl" width="608" height="402" allowfullscreen webkitallowfullscreen mozAllowFullScreen allow="autoplay *; fullscreen *; encrypted-media *" sandbox="allow-forms allow-same-origin allow-scripts allow-top-navigation allow-pointer-lock allow-popups allow-modals allow-orientation-lock allow-popups-to-escape-sandbox allow-presentation allow-top-navigation-by-user-activation" frameborder="0" title="Kaltura Player" style="position:absolute;top:0;left:0;width:100%;height:100%"></iframe></div></div>