# Logistic Regression

**Logistic regression** is the perceptron with one part swapped: the hard sign activation becomes a smooth **sigmoid**, so instead of committing to a class the neuron outputs a probability between 0 and 1. The cost changes to **binary cross-entropy**, but the update rule works out to the same $(\hat{y} - y)x$. Because it outputs probabilities, it handles overlapping data the perceptron couldn't separate.

This notebook implements one in NumPy and uses it to predict the probability that a breast tumor is malignant.

---

## How It Works

**Net input** — a weighted sum of the features plus a bias:

$$z = \mathbf{w} \cdot \mathbf{x} + b$$

**Activation (sigmoid)** — squashes the net input into a probability in $(0, 1)$:

$$\hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}}$$

**Learning rule** — for each example, the weights and bias step down the gradient of the cost:

$$w_i \leftarrow w_i - \eta(\hat{y} - y)x_i, \qquad b \leftarrow b - \eta(\hat{y} - y)$$

where $\eta$ is the learning rate. The gradient of cross-entropy through the sigmoid collapses to this same $(\hat{y} - y)x$ form, so the training loop is identical to the perceptron's.

---

## The Cost Function

$$C(\mathbf{w}, b) = -\frac{1}{N}\sum_{i=1}^{N}\big[\,y^{(i)}\log\hat{y}^{(i)} + (1 - y^{(i)})\log(1 - \hat{y}^{(i)})\,\big]$$

Binary cross-entropy rewards confident-and-correct probabilities and punishes confident-and-wrong ones. Minimizing it with gradient descent gives the update rule above.

---

## This Implementation

- **Dataset:** [breast cancer dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-wisconsin-diagnostic-dataset), loaded via scikit-learn or from `../Datasets/breast_cancer.csv`. Labels are remapped so that 1 = malignant.
- **Task:** predict the **probability a tumor is malignant**, first from a single feature (**worst concave points**), then from two (adding **worst radius**).
- **Model:** a `LogisticRegression` class (`net_input`, `predict`, `train`) that applies the sigmoid and tracks cross-entropy per epoch.
- **Result:** about 89% test accuracy on one feature, rising to about 95% on two, with the errors landing in the overlap region where the uncertainty is genuine.

The notebook walks through loading and encoding the data, visualizing the overlap, implementing the model, training, plotting the learned **S-curve**, evaluating with accuracy and a confusion matrix, and then extending to two features with a 3D probability surface, with interactive cells to test your own inputs.

---

## Limitations

- **Linear decision boundary** — it handles overlap with probabilities, but it still separates the classes with a straight line (a plane in higher dimensions), so genuinely curved boundaries need a more flexible model.
- **Needs a threshold** — the output is a probability, so turning it into a malignant/benign decision means choosing a cutoff (0.5 here); the right cutoff depends on how costly a false negative is versus a false positive.

---

## Files

- `Logistic-Regression.ipynb` — the full implementation and walkthrough.
- `../Datasets/breast_cancer.csv` — the dataset (also available through scikit-learn).