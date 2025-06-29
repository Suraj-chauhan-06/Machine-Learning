# -*- coding: utf-8 -*-
"""Regression.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1n_iQH-gb2iVevXrH8ApbJ4gxbQqX_Lc0

1. What is Simple Linear Regression
Simple Linear Regression is a method to model the relationship between a dependent variable and one independent variable using a straight line (Y = mX + c).


2. What are the key assumptions of Simple Linear Regression
Linearity

Independence

Homoscedasticity

Normality of residuals


3. What does the coefficient m represent in the equation Y = mX + c
It represents the slope, indicating the change in Y for a one-unit increase in X.



4. What does the intercept c represent in the equation Y = mX + c
It is the expected value of Y when X is zero.


5. How do we calculate the slope m in Simple Linear Regression
𝑚
=
∑
(
𝑋
𝑖
−
𝑋
ˉ
)
(
𝑌
𝑖
−
𝑌
ˉ
)
∑
(
𝑋
𝑖
−
𝑋
ˉ
)
2
m=
∑(X
i
​
 −
X
ˉ
 )
2

∑(X
i
​
 −
X
ˉ
 )(Y
i
​
 −
Y
ˉ
 )
​





6. What is the purpose of the least squares method in Simple Linear Regression
To minimize the sum of squared differences between actual and predicted values.



7. How is the coefficient of determination (R²) interpreted in Simple Linear Regression
It represents the proportion of variance in Y explained by X. R² ranges from 0 to 1.



8. What is Multiple Linear Regression
A method to model the relationship between a dependent variable and multiple independent variables.



9. What is the main difference between Simple and Multiple Linear Regression
Simple uses one independent variable, while Multiple uses two or more.




10. What are the key assumptions of Multiple Linear Regression
Linearity

Independence

Homoscedasticity

Normality of residuals

No multicollinearity






11. What is heteroscedasticity, and how does it affect the results of a Multiple Linear Regression model
Heteroscedasticity means unequal variance in residuals. It leads to inefficient and biased estimates.




12. How can you improve a Multiple Linear Regression model with high multicollinearity
Remove or combine highly correlated variables

Use PCA (Principal Component Analysis)

Use regularization (Ridge/Lasso)



13. Common techniques for transforming categorical variables
One-hot encoding

Label encoding

Binary encoding




14. What is the role of interaction terms
They model the combined effect of two variables on the outcome.




15. How does interpretation of intercept differ between Simple and Multiple Linear Regression
In Simple: Y when X = 0

In Multiple: Y when all X variables = 0




16. What is the significance of the slope?
It shows the strength and direction of the relationship.



17. How does the intercept provide context?
It provides the baseline Y value when all predictors are zero.





18. Limitations of using R² alone
Doesn't indicate causation

Can be misleading (overfitting)

Doesn’t show model validity




19. Interpretation of large standard error for a coefficient
Means the coefficient estimate is unreliable or imprecise.



20. How to identify heteroscedasticity in residual plots?
Residuals show a funnel or fan shape

Important to correct it for valid statistical inference




21. High R² but low adjusted R²?
Possible overfitting — adding unnecessary variables.




22. Why scale variables in Multiple Linear Regression?
To ensure equal contribution of features and help with model convergence.




23. What is polynomial regression?
Regression where predictors are raised to powers > 1.




24. How does it differ from linear regression?
Polynomial fits curves, not just lines.





25. When is polynomial regression used?
When the relationship is non-linear.




26. General equation for polynomial regression
𝑌
=
𝑏
0
+
𝑏
1
𝑋
+
𝑏
2
𝑋
2
+
.
.
.
+
𝑏
𝑛
𝑋
𝑛
Y=b
0
​
 +b
1
​
 X+b
2
​
 X
2
 +...+b
n
​
 X
n




27. Can it be applied to multiple variables?
Yes — Multivariate Polynomial Regression.




28. Limitations of polynomial regression
Prone to overfitting

Computationally intensive

Poor extrapolation





29. Methods to evaluate polynomial degree
Cross-validation

Adjusted R²

AIC/BIC





30. Why is visualization important?
To detect overfitting and see how well the curve fits.
"""

#31. Python code for Polynomial Regression

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
model.fit(X, y)