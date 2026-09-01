# Linear Regression From Scratch

This repository contains **Simple Linear Regression** and **Multiple Linear Regression** implemented completely from scratch using Python and NumPy.

The goal of this project is to understand how linear regression works internally rather than relying directly on machine-learning libraries such as `scikit-learn`.

## 📌 Contents

- Simple Linear Regression from scratch
- Multiple Linear Regression from scratch
- Parameter estimation using the **Normal Equation**
- Prediction using learned coefficients
- Calculation of the intercept and weights
- Model evaluation
- Comparison with library-based implementations

---

## 🧠 1. Simple Linear Regression

Simple Linear Regression is used when there is:

- **One independent variable** `X`
- **One dependent variable** `y`

The model is represented as:

$$
\hat{y} = c + mx
$$

Where:

- `c` = intercept
- `m` = weight/coefficient of the feature
- `x` = input feature
- `ŷ` = predicted value

---

## 🧮 2. Multiple Linear Regression

Multiple Linear Regression extends Simple Linear Regression to multiple independent variables.

For `n` features:

$$
\hat{y} =
b_0+b_1x_1+b_2x_2+\dots+b_nx_n
$$

Here:

- `b₀` = intercept
- `b₁, b₂, ..., bₙ` = weights/coefficient of the features
- `x₁, x₂, ..., xₙ` = input features

### Matrix Representation

The model can be represented as:

$$
\hat{y}=X\beta
$$

To include the intercept, a column of ones is added to `X`:

$$
X =
\begin{bmatrix}
1 & x_{11} & x_{12} & \dots & x_{1n}\\
1 & x_{21} & x_{22} & \dots & x_{2n}\\
\vdots & \vdots & \vdots & \vdots & \vdots\\
1 & x_{m1} & x_{m2} & \dots & x_{mn}
\end{bmatrix}
$$

The coefficients can be obtained using the **Normal Equation**:

$$
\beta = (X^TX)^{-1}X^Ty
$$

The resulting vector contains:

$$
\beta =
\begin{bmatrix}
b_0\\
b_1\\
b_2\\
\vdots\\
b_n
\end{bmatrix}
$$

So the model's **weights can be directly obtained** from the coefficient vector.

---

## 📊 Model Evaluation

The implementations can be evaluated using common regression metrics such as:

### R² Score

$$
R^2 =
1-
\frac{\sum(y_i-\hat{y}_i)^2}
{\sum(y_i-\bar{y})^2}
$$

`R²` describes how much of the variation in the target variable is explained by the model.

- `R² = 1` → perfect fit
- `R² = 0` → model performs like predicting the mean
- `R² < 0` → model performs worse than the mean baseline

---

## 🚀 Why Build Linear Regression From Scratch?

Implementing the algorithm from scratch helps understand:

- How regression coefficients are calculated
- How the intercept is obtained
- What model weights actually represent
- How matrix operations are used in machine learning
- How predictions are generated
- How regression metrics are calculated
- What happens internally when using a library implementation

Instead of treating `LinearRegression()` as a black box, this project focuses on understanding the mathematics behind it.

---

## 🛠️ Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn

No machine-learning library is required for the core regression implementation.

---

## 🔍 Simple vs Multiple Linear Regression

| Feature | Simple Linear Regression | Multiple Linear Regression |
|---|---|---|
| Number of features | 1 | 2 or more |
| Equation | `ŷ = b₀ + b₁x` | `ŷ = b₀ + b₁x₁ + ... + bₙxₙ` |
| Main parameters | 1 weight + intercept | Multiple weights + intercept |
| Matrix operations | Optional | Commonly used |
| Use case | One feature predicts target | Multiple features predict target |

---

## 📚 Concepts Practiced

- Simple Linear Regression
- Multiple Linear Regression
- Independent and dependent variables
- Intercept and coefficients
- Normal Equation
- Matrix multiplication
- Transpose
- Matrix inverse
- Predictions
- R² Score
- Regression model interpretation

---

## ⚠️ Note

The Normal Equation is useful for learning and for relatively small problems, but directly computing a matrix inverse can become computationally expensive or numerically unstable for large datasets or highly correlated features.

In practical machine-learning systems, optimized numerical methods such as QR decomposition, SVD, or library implementations are generally preferred.

---

## 📈 Future Improvements

Possible extensions to this project include:

- Gradient Descent implementation
- Feature scaling
- Polynomial Regression
- Regularization (Ridge and Lasso)
- Train/test split
- Cross-validation
- Comparison with `sklearn.linear_model.LinearRegression`
- Visualization of regression lines and prediction errors

---

## 👨‍💻 Author

**Pushkar Sinha**

This project was created as a learning exercise to understand the mathematics and implementation of Linear Regression from the ground up.
