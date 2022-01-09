# Split–apply–combine

One of the most important workflows in data analysis is called **split–apply–combine**:

1. Split the data into groups based on a criterion (e.g., species, marital status).
2. Apply operations to the data within each group.
3. Combine the results from the groups.

Of these, the "apply" step is usually the most complex. 




There are three types of operations:

* **Aggregation** refers to summarizing data by a single value, such as a sum or mean, or by a few values, such as value counts or quintiles.
* **Transformation** refers to application of a mathematical operation to every data value, resulting in data indexed the same way as the original. For example, quantitative data might be transformed to lie in the interval $[0,1]$.
* **Filtration** refers to inclusion/removal of a group based on a criterion, such as rejection of a group with too few members.

In some cases, a **pipeline** of operations (i.e., mathematical composition of functions) might be applied.



