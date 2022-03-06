# Regression

Regression is the task of approximating the value of a dependent quantitative variable as a function of independent variables, sometimes called *predictors*. 

Regression and classification are distinct but not altogether different. Abstractly, both are concerned with finding a function $f$ whose domain is feature space. In classification, the range of $f$ is a finite set of class labels, while in regression, the range is the real number line, or an interval. However, we can take the output of a regression and round or otherwise quantize it to get a finite set of classes, so regression can be used for classification. Likewise, most classification methods have a generalization to regression.

In addition to prediction tasks, some regression methods can be used to identify the relative significance of each predictor feature, and whether is has a direct or inverse relationship to the function value. Unimportant features can be removed to help minimize overfitting. 