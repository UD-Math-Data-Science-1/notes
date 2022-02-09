# Exercises

For these exercises, you may of course use computer help to work on a problem, but your answer should be self-contained without reference to computer output (unless stated otherwise).

1. For $n>2$, Let $x_i=0$ for $i=1,\ldots,n-1$, and $x_n=M$. Find the **(a)** mean, **(b)** median, **(c)** variance $s_{n-1}^2$, and **(d)** z-scores of the samples in terms of $M$ and $n$.

2. Suppose any samples $x_1,\ldots,x_n$ have z-scores $z_1,\ldots,z_n$. 

    **(a)** Show that $\displaystyle \sum_{i=1}^n z_i = 0.$

    **(b)** Show that $\displaystyle \sum_{i=1}^n z_i^2 = n-1.$
 
3. For the sample set in Exercise 1, find a value $N$ such that if $n>N$, there is at least one outlier according to the 2σ criterion.

4. Define a population by

    $$
    x_i = \begin{cases}
    1, & 1 \le i \le 11, \\ 
    2, & 12 \le i \le 14,\\ 
    4, & 15 \le i \le 22, \\ 
    6, & 23 \le i \le 32.
    \end{cases}
    $$

    **(a)** Find the median of the population.

    **(b)** Which of the following are outliers according to the 1.5 IQR criterion?

    $$-5,0,5,10,15,20$$

5. Suppose that a population has values $x_1,x_2,\ldots,x_n$. Define the function 

    $$
    r_2(x) = \sum_{i=1}^n (x_i-x)^2.
    $$

    Show that $r_2$ has a global minimum at $x=\mu$, the population mean.

6. Suppose that $n=2k+1$ and a population has values $x_1,x_2,\ldots,x_{n}$ in sorted order, so that the median is equal to $x_k$. Define the function 

    $$
    r_1(x) = \sum_{i=1}^n |x_i-x|.
    $$

    Show that $r_1$ has a global minimum at $x=x_k$ by way of the following steps. 
    
    **(a)** Explain why the derivative of $r_1$ is undefined at every $x_i$. Consequently, all of the $x_i$ are critical points of $r_1$. 
    
    **(b)** Determine $r_1'$ within each piece of the real axis between observation values, and explain why there cannot be any more critical points to consider. (Note: you can drop all of the absolute values when $x$ between the $x_i$, if you write out the expression in a certain way.)

    **(c)** By appealing to the derivative, explain why it must be that

    $$
    r_1(x_1) \ge r_1(x_2) \ge \cdots \ge r_1(x_k) \le r_1(x_{k+1}) \le \cdots \le r_1(x_n).
    $$


7. Prove that two sample sets have a Pearson correlation coefficient equal to 1 if they have identical z-scores.

8. Suppose that two sample sets satisfy $y_i=-x_i$ for all $i$. Prove that the Pearson correlation coefficient between $x$ and $y$ equals $-1$.

