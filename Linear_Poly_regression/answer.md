Loss function used in Linear Regression?
- To measure how well the model performs, a loss function is used.
- The most commonly used loss function in linear regression is Mean Squared Error (MSE)

Why squared error is used?
- Firstly, Squaring the error ensures that positive and negative errors do not cancel each other.
- Secondly, Squaring the error penalizes the larger errors, which will lead to avoid large mistakes.

What minimizing the loss actually means?
- Finding the set of model parameters that makes the model's predictions as close as possible to the true values on average.
- When the MSE error is minimized, the overall discrepancy between predicted and actual values is reduced.

How model parameters influence the loss?
- The model parameters determine the shape and position of the regression line or curve.
- Changing these parameters changes the predicted and actual values also changes, which directly affects the loss.

Why does training error always decrease with higher polynomial degree?
- As the polynomial degree increses, the model becomes more flexiblle and can fit the training data more closely.
- Because of this, the model can always reduce or maintain the training error.
- which is clearly shown by the downward of the training error curve in the training vs testing error plot.

Why does test error behave differently?
- The test error intially decreases beacuse increasing the polynomial degree helps the model capture the true non-linear relationship in the data.
- However, after a certain point, the model becomes too complex and starts learning noise instead of the real patterns.
- This behavior between the training and test error is visible in the plot and that highlights the problem of "Overfitting".

At what point does the model start overfitting, and how can you tell?
- From the training vs testing error plot, overfitting starts at degree 4.
- At this point, the training error remains very low, but the test error stops decreasing and begins to increase.
- This gap between training & test error indicates that the model is fitting the training data too closely and is no longer generalizing well to new data.

Which polynomial degree would you choose, and why?
- The best polynomial degree to choose is degree 3.
- This degree has the lowest test error & produces a smooth curve that follows the overall data without fitting noise.
- The degree-3 model captures the cubic relationship well.
