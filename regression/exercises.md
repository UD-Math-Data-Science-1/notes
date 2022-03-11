# Exercises

1. Suppose that the distinct plane points $(x_i,y_i)$ for $i=1,\ldots,n$ are to be fit using a linear function without intercept, $f(x)=\alpha x$. Use calculus to find a formula for the value of $\alpha$ that minimizes the sum of squared residuals,
    
    $$ r = \sum_{i=1}^n (f(x_i)-y_i)^2. $$

2. Suppose that $x_1=-2$, $x_2=1$, and $x_3=2$. Define $\alpha$ as in Exercise 1, and define the predicted values $\hat{y}_k=\alpha x_k$ for $k=1,2,3$. Express each $\hat{y}_k$ as a combination of the three values $y_1$, $y_2$, and $y_3$, which remain arbitrary. (This is a special case of a general fact about linear regression: each prediction is a linear combination of the training values.)

3. Using the formulas derived in {numref}`section-regression-linear`, show that the point $(\bar{x},\bar{y})$ always lies on the linear regression line. (Hint: You only have to show that $f(\bar{x}) = \bar{y}$. This can be done without first solving for $a$ and $b$, which is a bit tedious to write out.)

4. Suppose that values $y_i$ for $i=1,\ldots,n$ are to be fit to features $(u_i,v_i)$ using a multilinear function $f(u,v)=\alpha u + \beta v$. Define the sum of squared residuals
    
    $$ r = \sum_{i=1}^n (f(u_i,v_i)-y_i)^2. $$

    Show that by holding $\alpha$ is constant and taking a derivative with respect to $\beta$, and then holding $\beta$ constant and taking a derivative with respect to $\alpha$, at the minimum residual we must have 

    $$
    \left(\sum u_i^2 \right) \alpha + \left(\sum u_i v_i \right) \beta &= \sum u_i y_i, \\ 
    \left(\sum u_i v_i \right) \alpha + \left(\sum v_i^2 \right) \beta &= \sum v_i y_i. 
    $$

5. Repeat Exercise 1, but using the regularized residual 

    $$ \tilde{r} = C \alpha^2 + \sum_{i=1}^n (f(x_i)-y_i)^2. $$

6. Repeat Exercise 3, but using the regularized residual 

    $$ \tilde{r} = C (\alpha^2 + \beta^2) + \sum_{i=1}^n (f(x_i)-y_i)^2. $$

7. The probability $p$ of winning a race is predicted to obey the fit $\logit(p)=\alpha x + \beta$. If $\alpha=3$, $\beta=-1$, what are the odds (odds ratio) of winning the race at $x=1$?