# Linear Regression From Scratch

A from-scratch implementation of **Simple Linear Regression** and **Multiple Linear Regression** using Python and NumPy.

The main purpose of this project is to understand what happens inside `LinearRegression()` — especially how the **bias/intercept** and **weights/coefficients** are calculated mathematically.

---

## 📌 What This Project Covers

- Simple Linear Regression from scratch
- Multiple Linear Regression from scratch
- Ordinary Least Squares (OLS)
- Cost / error function
- Derivation of the Normal Equation
- Calculation of model weights `β`
- Calculation of bias/intercept `β₀`
- Prediction using matrix multiplication
- R² score evaluation
- Comparison with `sklearn.linear_model.LinearRegression`

---

# 1. Simple Linear Regression

Simple Linear Regression uses one independent variable to predict the target.

The equation is:

\[
\hat{y} = b_0 + b_1x
\]

where:

- \(b_0\) = bias/intercept
- \(b_1\) = weight/slope
- \(x\) = input feature
- \(\hat{y}\) = predicted value

### Finding the weight

\[
b_1 =
\frac{\sum_{i=1}^{m}(x_i-\bar{x})(y_i-\bar{y})}
{\sum_{i=1}^{m}(x_i-\bar{x})^2}
\]

### Finding the bias

\[
b_0 = \bar{y} - b_1\bar{x}
\]

After finding \(b_0\) and \(b_1\), prediction is simply:

\[
\hat{y}=b_0+b_1x
\]

---

# 2. Multiple Linear Regression

Multiple Linear Regression extends the same idea to several input features.

For \(n\) features:

\[
\hat{y}
=
\beta_0+\beta_1x_1+\beta_2x_2+\cdots+\beta_nx_n
\]

In matrix form:

\[
\hat{y}=X\beta
\]

To include the intercept, a column of ones is added to \(X\):

\[
X =
\begin{bmatrix}
1 & x_{11} & x_{12} & \dots & x_{1n}\\
1 & x_{21} & x_{22} & \dots & x_{2n}\\
\vdots & \vdots & \vdots & \ddots & \vdots\\
1 & x_{m1} & x_{m2} & \dots & x_{mn}
\end{bmatrix}
\]

and

\[
\beta=
\begin{bmatrix}
\beta_0\\
\beta_1\\
\beta_2\\
\vdots\\
\beta_n
\end{bmatrix}
\]

Therefore:

\[
\hat{y}=X\beta
\]

---

# 3. Finding β Using OLS

This is the main mathematical calculation used in the Multiple Linear Regression implementation.

We start with the prediction:

\[
\hat{y}=X\beta
\]

The residual/error is:

\[
e=y-\hat{y}
\]

So:

\[
e=y-X\beta
\]

OLS chooses the values of \(\beta\) that minimize the **sum of squared errors**.

## Objective function

\[
E=(y-X\beta)^T(y-X\beta)
\]

Expand it:

\[
E=
(y^T-\beta^TX^T)(y-X\beta)
\]

Multiplying:

\[
E
=
y^Ty-y^TX\beta-\beta^TX^Ty+\beta^TX^TX\beta
\]

Since \(y^TX\beta\) is a scalar:

\[
y^TX\beta=\beta^TX^Ty
\]

Therefore:

\[
E=
y^Ty-2\beta^TX^Ty+\beta^TX^TX\beta
\]

---

## 4. Differentiate With Respect to β

Now differentiate the error with respect to \(\beta\):

\[
\frac{\partial E}{\partial\beta}
=
-2X^Ty+2X^TX\beta
\]

For the minimum error, set the derivative equal to zero:

\[
-2X^Ty+2X^TX\beta=0
\]

Move the first term to the other side:

\[
2X^TX\beta=2X^Ty
\]

Divide by 2:

\[
X^TX\beta=X^Ty
\]

These are called the **Normal Equations**.

---

# 5. Solve for β

Starting from:

\[
X^TX\beta=X^Ty
\]

Multiply both sides by \((X^TX)^{-1}\):

\[
(X^TX)^{-1}X^TX\beta
=
(X^TX)^{-1}X^Ty
\]

Since:

\[
(X^TX)^{-1}(X^TX)=I
\]

we get:

\[
\boxed{
\beta=(X^TX)^{-1}X^Ty
}
\]

This is the **Normal Equation** used in this project.

---

# 6. What Does β Contain?

The calculated vector is:

\[
\beta=
\begin{bmatrix}
\beta_0\\
\beta_1\\
\beta_2\\
\vdots\\
\beta_n
\end{bmatrix}
\]

The first value is the **bias/intercept**:

\[
\beta_0 = \text{bias}
\]

The remaining values are the **weights**:

\[
[\beta_1,\beta_2,\ldots,\beta_n]
=
\text{weights}
\]

So in code:

```python
self.b[0]   # bias / intercept
self.b[1:]  # weights / coefficients
```

This is also why the first column of \(X\) is filled with ones.

---

# 7. Prediction

Once \(\beta\) has been calculated:

\[
\hat{y}=X\beta
\]

For a new dataset, the same intercept column is added:

```python
x = np.insert(x, 0, 1, axis=1)
return x.dot(self.b)
```

So the matrix multiplication automatically performs:

\[
\beta_0+\beta_1x_1+\beta_2x_2+\cdots+\beta_nx_n
\]

---

# 8. From-Scratch Multiple Linear Regression

The core implementation follows the OLS derivation above:

```python
class MyLr:
    def fit(self, x, y):
        x = np.insert(x, 0, 1, axis=1)
        self.b = np.linalg.inv(x.T.dot(x)).dot(x.T.dot(y))

    def predict(self, x):
        x = np.insert(x, 0, 1, axis=1)
        return x.dot(self.b)
```

The important line is:

```python
self.b = np.linalg.inv(x.T.dot(x)).dot(x.T.dot(y))
```

which is exactly:

\[
\boxed{\beta=(X^TX)^{-1}X^Ty}
\]

The code therefore directly implements the mathematical derivation.

---

# 9. Dataset and Experiment

The Multiple Linear Regression notebook uses the **Diabetes dataset** available through `sklearn.datasets.load_diabetes()`.

The data is split into:

```python
x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=42
)
```

The notebook first trains:

```python
LinearRegression()
```

and then trains the custom `MyLr` class using the Normal Equation.

---

# 10. Comparison With Scikit-Learn

The custom implementation was compared with `sklearn.linear_model.LinearRegression`.

### R² score

The notebook obtained:

```text
R2 score of sklearn lr 0.4526027629719198
R2 score of mylr      0.4526027629719195
```

The values are effectively identical, showing that the custom implementation reaches the same OLS solution for this experiment.

### Learned weights

The notebook obtained:

```text
[  37.90402135 -241.96436231  542.42875852  347.70384391
 -931.48884588  518.06227698  163.41998299  275.31790158
  736.1988589    48.67065743]
```

### Learned bias

```text
151.34560453985995
```

The custom implementation produced the same weights and bias up to floating-point precision.

This gives a useful way to see that the values returned by a library model are not mysterious — they are the coefficients obtained by solving the least-squares problem.

---

# 11. R² Score

The model is evaluated using the coefficient of determination:

\[
R^2 =
1-
\frac{\sum(y_i-\hat{y}_i)^2}
{\sum(y_i-\bar{y})^2}
\]

A value closer to 1 indicates that the model explains more of the variation in the target.

For this experiment:

\[
R^2 \approx 0.453
\]

for both the scikit-learn and from-scratch implementations.

---

# 12. Simple vs Multiple Linear Regression

| Feature | Simple Linear Regression | Multiple Linear Regression |
|---|---|---|
| Number of features | 1 | Multiple |
| Equation | \(y=b_0+b_1x\) | \(y=\beta_0+\beta_1x_1+\cdots+\beta_nx_n\) |
| Weights | One | Multiple |
| Bias | One intercept | One intercept |
| Matrix form | Optional | Natural representation |
| OLS Normal Equation | Can be used | Used here |

---

# 13. Libraries Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

The actual regression calculation for the custom model is performed with **NumPy**, using matrix operations and the Normal Equation.

---

# 14. Project Structure

```text
Linear-Regression/
│
├── Simple_Linear_Regression/
│   └── ...
│
├── Multiple_Linear_Regression_OLS.ipynb
│
└── README.md
```

Adjust the folder names according to the final repository structure.

---

# 15. Key Takeaways

The most important part of this project is understanding the connection between the mathematics and the code:

\[
\boxed{\beta=(X^TX)^{-1}X^Ty}
\]

becomes:

```python
self.b = np.linalg.inv(X.T.dot(X)).dot(X.T.dot(y))
```

and:

\[
\hat{y}=X\beta
\]

becomes:

```python
X.dot(self.b)
```

The first element of \(\beta\) is the **bias/intercept**, while the remaining elements are the **weights**.

This makes it clear how Linear Regression learns its coefficients instead of treating the algorithm as a black box.

---

## ⚠️ Note on the Normal Equation

The implementation uses an explicit matrix inverse:

```python
np.linalg.inv(X.T @ X)
```

This is excellent for understanding the mathematics, but in practical applications it is generally better to use numerically stable approaches such as `np.linalg.solve`, QR decomposition, SVD, or optimized library implementations.

---

## 🚀 Future Improvements

- Implement Linear Regression using Gradient Descent
- Add Mean Squared Error from scratch
- Add train/test evaluation from scratch
- Visualize predictions and residuals
- Compare Gradient Descent with the Normal Equation
- Add polynomial regression
- Add Ridge and Lasso regression
