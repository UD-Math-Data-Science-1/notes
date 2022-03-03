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

where $w_1,w_2,b$ are constants. It is clear that $w_1 x_1 + w_2 x_2 + b > 0$ represents the half-plane on one side of the line, and $w_1 x_1 + w_2 x_2 + b < 0$ represents the other. 

Our next observation is that if the point $(a_1,a_2)$ is on the line, then any point of the form

$$
x_1 = a_1 - tw_2, \quad x_2 = a_2 + tw_1
$$ 

is also on the line. (Just substitute it in to see that it satisfies the equation of the line.)

Now let us find the distance from any point $(s_1,s_2)$ to the line. The distance squared from this point to an arbitrary line point is 

$$
d^2 = (s_1-a_1+tw_2)^2 + (s_2-a_2-tw_1)^2.
$$

Using calculus to minimize $d^2$ as a function of $t$ eventually gives

$$
t_\text{min} = \frac{(s_2-a_2)w_1-(s_1-a_1)w_2}{w_1^2+w_2^2}.
$$

Note that $w_1^2+w_2^2=\norm{\bfw}_2^2$. Substituting $t_\text{min}$ into $d^2$ and taking a square root gives the distance from $(s_1,s_2)$ to the line:

$$
d_\text{min} = \frac{|(s_1-a_1)w_1 + (s_2-a_2)w_2|}{\twonorm{\bfw}} = \frac{|s_1w_1 + s_2w_2 + b|}{\twonorm{\bfw}}.
$$

Suppose we use $y_i=+1$ for all the labels on one side, and $y_i=-1$ for all the labels on the other side. Finally, the condition that the distance from the line to point $(x_{i,1},x_{i,2})$ be no smaller than the margin $M$, and that the point be on the correct side of the line, is 

```{math}
:label: eq-svm-constraints2d
y_i\left( \frac{ x_{i,1}w_1 + x_{i,2}w_2 + b }{\norm{\bfw}_2} \right) \ge M,
```

which must hold true for all $i$.

Recall that the goal is to separate the two sets of points by a line as robustly as possible. Hence we want to maximize the margin $M$ while obeying the constraints implied by {eq}`eq-svm-constraints2d`. This is a *constrained optimization* problem. Algorithms for solving it are beyond our scope here.

## Higher dimensions

What happens in $d>2$ dimensions? Instead of a line, we have a **plane** ($d=3$) or **hyperplane** ($d>3$) of dimension $d-1$. Its equation can be expressed as

$$
w_1 x_1 + w_2 x_2 + \cdots + w_d x_d + b = 0,
$$

for some constants $w_1,\ldots,w_d,b$. The vector $\bfw=[w_1,\ldots,w_d]$ is said to be **normal** to the hyperplane. In 2D or 3D this is equivalent to being perpendicular to the plane.

We have the important new operation

$$
\bfw^T\bfx = w_1 x_1 + w_2 x_2 + \cdots + w_d x_d, 
$$

the **inner product** between vectors $\bfx$ and $\bfw$. (In 2D or 3D this is the same as the *dot product*.) It follows easily that

$$
\bfw^T\bfw = \twonorm{\bfw}^2,
$$

which is one fact that makes the 2-norm special. The (signed) distance from any point $\mathbf{s}=(s_1,\ldots,s_d)$ to the hyperplane $\bfw^T\bfx+b=0$ is 

$$
\frac{\bfw^T\mathbf{s}+b}{\twonorm{\bfw}}.
$$

Above we saw that $\twonorm{\bfw}$ is inversely related to the margin $M$. One form of the constrained optimization problem, known as the *primal formulation*, is

$$
\text{minimize } & \twonorm{\bfw}, \\ 
\text{subject to } & y_i(\bfw^T \bfx_i + b) \ge 1,\, i = 1,\ldots,n.
$$

Usually, though, the optimization is actually performed on an equivalent *dual formulation*, which finds the $d+1$ support vectors and the margin rather than $\bfw$ and $b$ directly.

## Advanced aspects

There are other refinements too advanced for us to go into in detail here. Two stand out for making the algorithm practical for more than a trivial number of problems. 

One is the idea of allowing *slack*, which means that points are allowed to be on the wrong side of the dividing hyperplane, but they are punished by an amount proportional to their distance from it. The balance between maximizing margin and punishing miscreants is controlled by a hyperparameter that is usually called $C$, and the algorithm may be called **C-SVM**.

The other important refinement is to upgrade the separating hyperplane to allow other kinds of surfaces. For reasons we won't go into, this is called the **kernel trick**, and specifying the kernel is another option. The most common choices are *linear*, which is the original hyperplane as described above, and *RBF*, which has its own hyperparameter $\gamma$ and can produce arbitrary decision boundaries.

## Usage in sklearn

Let's explore the usage of SVM with a dataset derived from images of breast tissue. The target classification is benign/malignant.

```{code-cell}
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer(as_frame=True)["frame"]
y = cancer["target"]
X = cancer.drop("target",axis=1)
print(sum(y==0),"malignant and",sum(y==1),"benign samples")

from sklearn.model_selection import train_test_split
X_tr, X_te, y_tr, y_te = train_test_split(X,y,test_size=0.2,shuffle=True,random_state=0)
```

The `svm` module in sklearn defines `SVC` for classification. By default, it uses an RBF kernel, but here we require it to use the linear kernel (that is, a true hyperplane as the decision boundary).

```{code-cell}
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

svc = SVC(C=1e-3,kernel="linear")
svc.fit(X_tr,y_tr)

yhat = svc.predict(X_te)
print("confusion matrix on test samples:")
print(confusion_matrix(y_te,yhat))
```

We can query the classifier for the normal vector $\bfw$ and value $b$ in the formulas above. 

```{code-cell}
w = svc.coef_[0]
b = svc.intercept_[0]
print("w:",w," b:",b)
```

The sign of $\bfw^T\bfx + b$ determines which side of the hyperplane a point $\bfx$ lies on. For example, the predictions

```{code-cell}
Xq = X_te.iloc[20:25,:]
print( svc.predict(Xq) )
```

correspond to the signs of the following:

```{code-cell}
def dot(u,v):
    return sum(u[i]*v[i] for i in range(len(u)))

Xq = Xq.to_numpy()
[dot(w,x)+b for x in Xq]
```

The training accuracy reflects the amount of slack allowed. If it's 100%, then every training observation lies on its proper side. 

```{code-cell}
print("train accuracy:",svc.score(X_tr,y_tr))
```

If we increase $C$, we penalize the slack more and increase the training accuracy. (That might or might not increase the test accuracy as well; we explore that relationship more in the next section.)

```{code-cell}
svc = SVC(C=1,kernel="linear")
svc.fit(X_tr,y_tr)
print("train accuracy with less slack:",svc.score(X_tr,y_tr))

yhat = svc.predict(X_te)
print("confusion matrix with less slack:")
print(confusion_matrix(y_te,yhat))
```

In general, the RBF kernel might be much better than the linear one. But there are no guarantees that it will be so, and the best value for $C$ might change.

```{code-cell}
svc = SVC(C=100)
svc.fit(X_tr,y_tr)
print("train accuracy with RBF kernel:",svc.score(X_tr,y_tr))

yhat = svc.predict(X_te)
print("confusion matrix with RBF kernel:")
print(confusion_matrix(y_te,yhat))

```

An SVM usually benefits from standardizing the features.

```{code-cell}
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler   

svc = make_pipeline(StandardScaler(),SVC(C=0.5))
svc.fit(X_tr,y_tr)

yhat = svc.predict(X_te)
print("confusion matrix with RBF and standardization:")
print(confusion_matrix(y_te,yhat))
```

<div style="max-width:400px"><div style="position:relative;padding-bottom:71.25%"><iframe id="kaltura_player" src="https://cdnapisec.kaltura.com/p/2358381/sp/235838100/embedIframeJs/uiconf_id/43030021/partner_id/2358381?iframeembed=true&playerId=kaltura_player&entry_id=1_eig901cz&flashvars[streamerType]=auto&amp;flashvars[localizationCode]=en&amp;flashvars[leadWithHTML5]=true&amp;flashvars[sideBarContainer.plugin]=true&amp;flashvars[sideBarContainer.position]=left&amp;flashvars[sideBarContainer.clickToClose]=true&amp;flashvars[chapters.plugin]=true&amp;flashvars[chapters.layout]=vertical&amp;flashvars[chapters.thumbnailRotator]=false&amp;flashvars[streamSelector.plugin]=true&amp;flashvars[EmbedPlayer.SpinnerTarget]=videoHolder&amp;flashvars[dualScreen.plugin]=true&amp;flashvars[Kaltura.addCrossoriginToIframe]=true&amp;&wid=1_x7zp0km7" width="400" height="285" allowfullscreen webkitallowfullscreen mozAllowFullScreen allow="autoplay *; fullscreen *; encrypted-media *" sandbox="allow-forms allow-same-origin allow-scripts allow-top-navigation allow-pointer-lock allow-popups allow-modals allow-orientation-lock allow-popups-to-escape-sandbox allow-presentation allow-top-navigation-by-user-activation" frameborder="0" title="Kaltura Player" style="position:absolute;top:0;left:0;width:100%;height:100%"></iframe></div></div>

<div style="max-width:400px"><div style="position:relative;padding-bottom:71.25%"><iframe id="kaltura_player" src="https://cdnapisec.kaltura.com/p/2358381/sp/235838100/embedIframeJs/uiconf_id/43030021/partner_id/2358381?iframeembed=true&playerId=kaltura_player&entry_id=1_sdqvba7v&flashvars[streamerType]=auto&amp;flashvars[localizationCode]=en&amp;flashvars[leadWithHTML5]=true&amp;flashvars[sideBarContainer.plugin]=true&amp;flashvars[sideBarContainer.position]=left&amp;flashvars[sideBarContainer.clickToClose]=true&amp;flashvars[chapters.plugin]=true&amp;flashvars[chapters.layout]=vertical&amp;flashvars[chapters.thumbnailRotator]=false&amp;flashvars[streamSelector.plugin]=true&amp;flashvars[EmbedPlayer.SpinnerTarget]=videoHolder&amp;flashvars[dualScreen.plugin]=true&amp;flashvars[Kaltura.addCrossoriginToIframe]=true&amp;&wid=1_dcp78qll" width="400" height="285" allowfullscreen webkitallowfullscreen mozAllowFullScreen allow="autoplay *; fullscreen *; encrypted-media *" sandbox="allow-forms allow-same-origin allow-scripts allow-top-navigation allow-pointer-lock allow-popups allow-modals allow-orientation-lock allow-popups-to-escape-sandbox allow-presentation allow-top-navigation-by-user-activation" frameborder="0" title="Kaltura Player" style="position:absolute;top:0;left:0;width:100%;height:100%"></iframe></div></div>
