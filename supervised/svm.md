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

# Support vector machine

Suppose for a moment that we have data samples with just two dimensions (features), and that they are arranged like this:

```{figure} ../_static/svm_many.png
```

These groups are easy to separate! In fact, as the figure shows, we are spoiled for choice, and the possibilities span a wide range, even if we use only straight lines. 

However, there is a way to define the *best* line separating the two sets, as illustrated in this figure:

```{figure} ../_static/svm_margins.png
```

The key is to define the **margin** between the sets as the maximum perpendicular distance between the points and the line. It turns out that, if the sets can be separated by a line at all, then there is a line that maximizes the margin. It's also the case that only a few of the sample points actually matter, as shown by the boxes in the figure. They are the ones that achieve the margins and are called **support vectors**. A learner based on this idea of separating data by maximum margin is called a **support vector machine** (SVM).

Let's express the line and margin mathematically in two dimensions. We are used to writing a line as $y=mx+b$. First, though, we are going to use subscripts rather than letters for the dimensions, so make that $x_2=mx_1+b$. Next, we recall that this equation doesn't work for vertical lines (infinite slope), so we need a coefficient in front of $x_2$ as well. Rearranging, we get 

$$
w_1 x_1 + w_2 x_2 + b = 0,
$$

where $w_1,w_2,b$ are constants. 

Our next observation is that if the point $(a_1,a_2)$ is on the line, then any point of the form

$$
x_1 = a_1 - tw_2, \quad x_2 = a_2 + tw_1
$$ 

is also on the line. (Just substitute it in to see that it satisfies the equation of the line.)

Now let us find the distance from any point $(s_1,s_2)$ to the line. The distance squared from this point to an arbitrary line point is 

$$
d^2 = (s_1-a_1+tw_2)^2 + (s2-a_2-tw_1)^2.
$$

Using calculus to minimize $d^2$ as a function of $t$ eventually gives

$$
t_\text{min} = \frac{(s_2-a_2)w_1-(s_1-a_1)w_2}{w_1^2+w_2^2}.
$$

Note that $w_1^2+w_2^2=\norm{[w_1,w_2]}_2^2$. Substituting $t_\text{min}$ into $d^2$ and taking a square root gives

$$
d_\text{min} = \frac{|(s_1-a_1)w_1 + (s_2-a_2)w_2|}{\norm{\bfw}^2}
 = \frac{|(s_1w_1 + s_2w_2 + b|}{\norm{\bfw}_2}.
$$

It is clear that $w_1 s_1 + w_2 s_2 + b > 0$ represents the half-plane on one side of the line, and $$w_1 s_1 + w_2 s_2 + b < 0$ represents the other. Suppose we use $y_i=+1$ for all the labels on one side, and $y_i=-1$ for all the labels on the other side. Finally, the condition that the distance from the line to point $(x_{i,1},x_{i,2})$ be no smaller than the margin $M$, and that the point be on the correct side of the line, is 

$$
y_i\left( \frac{ s_1w_1 + s_2w_2 + b }{\norm{\bfw}_2} \right) \ge M,
$$

which must hold true for all $i$ as we maximize the margin $M$. 

This is a *constrained optimization* problem. The details of how it's solved are interesting, but beyond us in this space.

## Higher dimensions

What happens in $d>2$ dimensions? Instead of a line, we have a **hyperplane** of dimension $d-1$. Its equation can be expressed as

$$
w_1 x_1 + w_2 x_2 + \cdots + w_d x_d + b = 0,
$$

for some constants $w_1,\ldots,w_d,b$. In fact, the vector $\bfw=[w_1,\ldots,w_d]$ is said to be perpendicular or **normal** to the hyperplane. 

We have the important new notation

$$
\bfw^T\bfx = w_1 x_1 + w_2 x_2 + \cdots + w_d x_d 
$$

as the **inner product** between vectors $\bfx$ and $\bfw$. It follows easily that

$$
\bfw^T\bfw = \norm{\bfw}_2^2,
$$

which is the important fact that makes the 2-norm special. One form of the constrained optimization problem (known as the *primal formulation*) is

$$
\text{minimize } & \norm{\bfw}_2 \\ 
\text{subject to } & y_i(\bfw^T \bfx_i + b) \ge 1,\, i = 1,\ldots,n.
$$

Usually, though, the optimization is actually performed on an equivalent *dual formulation*, which finds the $d+1$ support vectors and the margin rather than $\bfw$ and $b$ directly.

## Advanced aspects

There are other refinements too advanced for us to go into in detail here. Two stand out for making the algorithm practical for more than a trivial number of problems. 

One is the idea of allowing *slack*, which means that points are allowed to be on the wrong side of the dividing hyperplane, but they are punished by an amount proportional to their distance from it. The balance between maximizing margin and punishing miscreants is controlled by a hyperparameter that is usually called $C$, and the algorithm may be called **C-SVM**.

The other important refinement is to upgrade the separating hyperplane to allow other kinds of surfaces. For reasons we won't go into, this is called the **kernel trick**, and specifying the kernel is another option. The most common choices are *linear*, which is the original hyperplane, and *RBF*, which has its own hyperparameter $\gamma$.

## Usage in sklearn

The SVM classifier in sklearn is called `SVC`. By default, it uses $C=1$ and the RBF kernel.

```{code-cell}
import numpy as np
X = np.loadtxt("data.csv",delimiter=",")
y = np.loadtxt("labels.csv",delimiter=",")

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

X_tr, X_te, y_tr, y_te = train_test_split(X,y,test_size=0.2)

svc = svm.SVC()
svc.fit(X_tr,y_tr)

print("train accuracy:",svc.score(X_tr,y_tr))
print("test accuracy:",svc.score(X_te,y_te))
```

The training accuracy essentially tells us how much slack was allowed; i.e., how frequently sample points end up on the wrong side of the decision boundary. If we increase $C$, we penalize the slack more and increase the training accuracy. (That might increase the test error as well, but we explore that relationship more in the next section.)

```{code-cell}
svc = svm.SVC(C=100)
svc.fit(X_tr,y_tr)

print("train accuracy with less slack:",svc.score(X_tr,y_tr))
print("test accuracy with less slack:",svc.score(X_te,y_te))

yhat = svc.predict(X_te)
print(confusion_matrix(y_te,yhat))
```

SVM usually benefits from standardizing the features, so it's a good idea to build that in.

```{code-cell}
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler   

svc = make_pipeline(StandardScaler(),svm.SVC())
svc.fit(X_tr,y_tr)

print("train accuracy with standardization:",svc.score(X_tr,y_tr))
print("test accuracy with standardization:",svc.score(X_te,y_te))
```

