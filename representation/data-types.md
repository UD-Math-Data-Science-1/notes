# Types of data

Each source of data we want to work with has to be expressed mathematically, and ultimately, in digital form. 

## Single values

### Quantitative

A **quantitative** value is one that is numerical and supports meaningful comparison and arithmetic operations. Quantitative data is further divided into **continuous** and **discrete** types. The difference is the same as between real numbers and integers. 

```{prf:example}
Some continuous quantitative data sources:

* Temperature at noon at a given airport
* Your height
* Voltage across the terminals of a battery

Examples of discrete quantitative data:

* The number of shoes you own
* Number of people at a restaurant table
* Score on an exam

In each case, it makes sense, for example, to order different values and compute averages of them. (Note that averages of discrete quantities are continuous, though.)

Sometimes there may be room for interpretation or context. For example, the retail price of a gallon of milk might be regarded as discrete data, since it technically represents a whole number of pennies. But in finance, transactions are regularly computed to much higher precision, so it may make more sense to interpret prices as continuous values.
```

```{prf:example}
Not all numerical values represent truly quantitative data. ZIP codes (postal codes) in the U.S. are 5-digit numbers, and while there is some pattern to how they are assigned, there is no meaningful interpretation of, say, ordering them or averaging them.
```

```{prf:example}
A particular quantitative type is a **timestamp**. As the name implies, it represents a moment in time. Conceptually timestamps are continuous, although in most applications we might be concerned with a resolution of days, for instance, and consider them to be discrete. Timestamps can be surprisingly tricky to work with, as they must account for time zones and different conventions for their textual representation around the world; arithmetic between timestamps may be further complicated by the varying number of days in the months and the presence of leap years. One approach to these complications is to convert all timestamps to *Unix time*, which is the number of seconds since January 1, 1970. 
```

Mathematically, the real and integer number sets are infinite, but that is not possible in a computer. Integers are represented exactly within some range that is determined by how many binary bits are dedicated. The computational analog of real numbers are **floating-point numbers**, or more simply, **floats**. These are bounded in range as well as discretized. The details are complicated, but essentially, the floating-point numbers have about 16 significant digits if 64 bits are used, which is almost always far more precision than real data offers.

There are two additional quasi-numerical values to be aware of as well. The value `Inf` stands for infinity. It's greater than every finite number, `Inf+1` is equal to `Inf`, and so on. However, in calculus, `Inf-Inf` is considered an indeterminate value and gives the result `NaN`, which stands for *Not a Number*.

### Qualitative

A **qualitative** value is one that is not quantitative. One important subtype is **categorical** data, in which the values are drawn from a finite, discrete set. This set further bifurcates into **ordinal** data, which support meaningful ordering comparisons, and **nominal** data, which is unordered.

```{prf:example}
Examples of ordinal categorical data:

* Seat class on a commercial airplane (e.g., economy, business, first)
* Letters of the alphabet

Nominal categorical data:

* Yes/No responses
* Marital status
* Make of a car

There are nuanced cases. For instance, letter grades are themselves ordinal categorical data. However, schools convert them to discrete quantitative data and then compute a continuous quantitative GPA.
```

Besides categorical data, there are **text fields**, images, videos—just about anything, in principle. 

## Collections of values

In many real-world situations, it makes sense to consider an ensemble of values as a single unit. For example, wind velocity requires three continuous values, corresponding to directions along coordinate axes. If we label these axes as **x**, **y**, and **z**, then we could describe the velocity as a **series** of three values that are **indexed** by the set $\{\bfx,\bfy,\bfz\}$. 

Index sets tend to be categorical or discrete data. For instance, you could have a series of light wavelengths indexed by color names. When the index set consists of timestamps, we have a **time series**. When the index set consists of consecutive integers, we describe the series as a **vector** or a **one-dimensional array**. If the indices are pairs of integers taken from consecutive sets, we have a **matrix** or a **two-dimensional array**.

A **data frame** is a collection of series that share the same index set. The individual series are called **columns** of the data frame, and the values of all the columns at a single index value form a **row**. 

## Dummy/one-hot encoding

One way to convert nominal data to a quantifiable form is through **dummy variables**, also called **one-hot encoding**. A category that has $m$ unique options $c_1,\ldots,c_m$ can be replaced by $m-1$ binary variables $x_i$. The value $c_1$ corresponds to $x_1=1$, $x_2=\cdots=x_{m-1}=0$. The value $c_2$ is translated into $x_2=1$, $x_1=x_3=\cdots=0$, and so on. The value $c_m$ means that all the $x_i$ are zero.

::::{prf:example}
:label: example-data-types-dummy
Suppose that the *stooge* variable can take the values `Moe`, `Larry`, `Curly`, or `Shemp`. Then the vector 

```
[Curly,Moe,Curly,Shemp]
```

would be replaced by the array

```
[ [0,0,1],[1,0,0],[0,0,1],[0,0,0] ]
```

(Note: Nobody likes Larry.)
::::

## Missing values

In real data sets, we often must cope with data series that have missing values. This is a common source of mistakes and confusion, especially because there is no universal practice. Sometimes zero is used to represent a missing number. It's also common to use an impossible value, such as $-999$ to represent weight, to signify missing data.

While a few computer languages have a dedicated "missing" value defined, in most cases (including Python) it's common to use `NaN` to represent a missing value, even when the rest of the data in the series are not expressed as float numbers. It's useful to know the following. 

```{warning}
By definition, every operation that involves a `NaN` value results in a `NaN`.
```

Notoriously, one consequence of this behavior is that `NaN==NaN` is `NaN`, not "true!" When you get unexpected or inexplicable behavior, it's worth determining whether you have been unknowingly operating with `NaN`s.

The process of detecting and dealing with missing or impossible values is part of **data cleaning**. (Other data cleaning tasks include dealing with incorrectly formatted input and removing duplicates.) Sometimes, the missing values are simply dropped from the data; in other cases, they are replaced with average or other representative values through a process called **imputation**. 


