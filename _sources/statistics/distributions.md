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
# Distributions

We will illustrate some concepts using a data set that is provided by seaborn, a useful data visualization package.

```{code-cell} ipython3
import pandas as pd
import seaborn as sns

cars = sns.load_dataset("mpg")
cars
```

## Histogram

One of the fastest ways to become familiar with a data set is to visualize it. To visualize a distribution of continuous values, we can use a **histogram**. We choose ordered real values $t_1 < t_2 < \cdots < t_m$. Define *bin* $k$ to be the interval $B_k=[t_k,t_{k+1})$, with the convention that $t_0=-\infty$ and $t_{m+1}=\infty$. This gives us a series of counts $c_k$, where $c_k$ is defined as the number of data values in interval $B_k$. A histogram plots those counts, typically using a bar graph.

In seaborn, you plot a histogram using `displot`.

```{code-cell}
sns.displot(data=cars,x="mpg");
```

You can specify the number of bins used (or their edges).

```{code-cell}
sns.displot(data=cars,x="mpg",bins=20);
```

Much as we use grouping in pandas, we can use categories in a series to split the data in seaborn. In this case, we use three different colors (hues) for three different regions of origin.

```{code-cell}
sns.displot(data=cars,x="mpg",hue="origin");
```

That graph is a little messy because of the overlaps. We can instead split the data and plot as columns of subplots.

```{code-cell}
sns.displot(data=cars,x="mpg",col="origin");
```

In either case, it's easy to see that the U.S. cars are more clustered on the left (smaller MPG) than the Japanese and European cars.

You can even combine grouping by plot column with grouping by color.

```{code-cell}
sns.displot(data=cars,x="mpg",col="origin",hue="cylinders",multiple="stack");
```


## Distribution functions

The **cumulative distribution function** (CDF) $F(x)$ is the probability that a data value will be less than or equal to $x$. The domain of $F$ is the real line (unless otherwise restricted), and the range is the interval $[0,1]$.

When a sample is used to estimate a continuous CDF, the resulting is a stair-step (piecewise constant) graph known as the **empirical CDF**.

```{code-cell}
sns.displot(data=cars,x="mpg",col="origin",kind="ecdf");
```

If we could account for infinitely many observations, we would expect (in most cases) the CDF to be continuous and differentiable. The derivative of the CDF, $f(x)=F'(x)$, is called the **probability distribution function** (PDF). The PDF is the continuous analog of a histogram divided into infinitesimally small bins. Note that because the probability of observing *some* real value is 100%, we have the constraint

$$
\int_{-\infty}^\infty f(x)\, dx = 1.
$$

## Normal distribution

The granddaddy of distributions is the **normal** or **Gaussian** distribution. It is parameterized by its mean $\mu$ and standard deviation $\sigma$, and has PDF

$$
f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left( -\frac{(x-\mu)^2}{2\sigma^2} \right).
$$

The CDF cannot be expressed using elementary functions. A normal PDF is peaked at the mean, dropping superexponentially and symmetrically on both sides.

```{code-cell}
:tags: ["hide-input"]
import numpy as np
import matplotlib.pyplot as plt

mu = 1
sigma = 2
x = mu + np.arange(-200,200)*12/400
s = np.sqrt(2*np.pi*sigma)
plt.plot(x,np.exp(-(x-mu)**2/(2*sigma*2))/s)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("PDF of normal distribution, mu=1, sigma=2");
```

## Kernel density estimation

It can be helpful to go from a discrete distribution, as described by a histogram, to a continuous function that approximates the PDF. One technique to do this is called **kernel density estimation** (KDE).

Let $g(x)$ be a normal distribution with mean zero and a chosen variance. If we have samples $x_1,\ldots,x_n$, define

$$
\tilde{f}(x) = \sum_{i=1}^n g(x-x_i).
$$

That is, we add up normal distributions centered at the observed values. Finally, we define

$$
\frac{\tilde{f}(x) }{\int_{-\infty}^\infty \tilde{f}(x) \, dx}
$$

as the KDE approximation to the PDF.

For example, using the MPG data:

```{code-cell}
sns.displot(data=cars,x="mpg",hue="origin",kind="kde");
```

Based on these curves, it would be plausible to approximate the distribution of cars from Europe as a normal distribution, but the asymmetry in the other two cases would make that approximation less satisfying.