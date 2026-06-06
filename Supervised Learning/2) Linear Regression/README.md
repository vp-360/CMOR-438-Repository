# Linear Regression

**Linear regression** is the same single-neuron model as the perceptron, with two parts swapped: a **linear activation** in place of the sign function, so the output is a real number instead of a class, and **mean squared error** in place of the misclassification count. It's trained the same way, with stochastic gradient descent.

This notebook implements one in NumPy and uses it to predict diabetes disease progression from a patient's BMI.

---

## How It Works

**Net input** — a weighted sum of the features plus a bias:

$$z = \mathbf{w} \cdot \mathbf{x} + b$$

**Activation (linear)** — the activation passes the net input through unchanged, so the prediction is just the net input:

$$\hat{y} = z$$

**Learning rule** — for each example, the weights and bias step down the gradient of the cost:

$$w_i \leftarrow w_i - \alpha(\hat{y} - y)\,x_i, \qquad b \leftarrow b - \alpha(\hat{y} - y)$$

where $\alpha$ is the learning rate. Because the activation is linear, the gradient is simple, and stochastic gradient descent applies this update one patient at a time.

---

## The Cost Function

$$C(\mathbf{w}, b) = \frac{1}{2N} \sum_{i=1}^{N} \left( \hat{y}^{(i)} - y^{(i)} \right)^2$$

The mean squared error averages the squared gap between each prediction and its true target. Squaring punishes large misses more, and the $\frac{1}{2}$ makes the derivative come out clean. Minimizing it with gradient descent gives the update rule(learning rule) above.

---

## This Implementation

- **Dataset:** [diabetes dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset), loaded via scikit-learn or from `../Datasets/diabetes.csv`.
- **Task:** predict **disease progression** one year after baseline from a single feature, **BMI**. One feature is used so the fit can be drawn as a line, though the model works with any number.
- **Model:** a `LinearRegression` class (`net_input`, `predict`, `train`) that tracks the MSE per epoch.
- **Result:** The model reaches a test RMSE of about 63 units. RMSE isn't the average error, it's the root of the mean of *squared* errors, so it's weighted toward the larger misses, but it puts the error in the target's own units. Because BMI alone explains just part of the variance in disease progression, so the model hits a ceiling no amount of extra training can break through. The test RMSE matches the training RMSE, confirming the model generalizes well, it's just limited by having only one input feature.

The notebook walks through loading the data, checking the relationship is linear, splitting into train and test and standardizing BMI, implementing the model, training with SGD, evaluating with RMSE, and plotting both the fitted line and the MSE convergence curve.

---

## Limitations

- **Straight lines only** — it assumes the relationship is linear and can't fit curved patterns. (The notebook checks the BMI–progression trend is roughly straight before fitting.)
- **A single feature has a ceiling** — BMI alone explains only part of disease progression, so the error settles at a nonzero floor (the irreducible noise) rather than reaching zero. Doing better would need more features, which points toward multiple linear regression.

---

## Files

- `Linear-Regression.ipynb` — the full implementation and walkthrough.
- `../Datasets/diabetes.csv` — the dataset (also available through scikit-learn).
