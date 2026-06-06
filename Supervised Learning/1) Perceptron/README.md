# The Perceptron

The **Perceptron** (Rosenblatt, 1958) is the simplest neural-network model: a single neuron that learns a straight-line boundary between two classes. It computes a weighted sum of its inputs, applies a sign activation, and nudges its weights whenever it misclassifies a point, repeating until every point is correct.

This notebook implements one in NumPy and uses it to classify two penguin species by their bill measurements.

---

## How It Works

**Net input** — a weighted sum of the features plus a bias:

$$z = \mathbf{w} \cdot \mathbf{x} + b$$

**Activation** — the sign function turns that sum into a label of $-1$ or $+1$:

$$\hat{y} = \text{sign}(z)$$

**Learning rule** — for each example, the weights and bias shift toward the correct answer:

$$w_i \leftarrow w_i + \frac{\eta}{2}(y - \hat{y})\,x_i, \qquad b \leftarrow b + \frac{\eta}{2}(y - \hat{y})$$

If the prediction is right, $y - \hat{y} = 0$ and nothing changes; if it's wrong, the weights move by an amount proportional to each feature's value, scaled by the learning rate $\eta$. The **Perceptron Convergence Theorem** guarantees this reaches zero errors if the data is linearly separable.

---

## The Cost Function

$$C(\mathbf{w}, b) = \frac{1}{4} \sum_{i=1}^{N} \left( \hat{y}^{(i)} - y^{(i)} \right)^2$$

With $\pm 1$ labels, $(\hat{y} - y)^2$ is either $0$ (correct) or $4$ (wrong), so the $\frac{1}{4}$ factor makes the cost equal the number of misclassified points. The $\frac{\eta}{2}$ in the update rule comes from minimizing this cost.

---

## This Implementation

- **Dataset:** [Palmer Penguins](https://allisonhorst.github.io/palmerpenguins/), loaded via seaborn or from `../Datasets/penguins.csv`.
- **Task:** classify **Adelie** ($-1$) vs. **Gentoo** ($+1$) using **bill length** and **bill depth** (mm). Two features are used so the decision boundary can be drawn as a line, though the perceptron works with any number.
- **Model:** a `Perceptron` class (`net_input`, `predict`, `train`) that tracks errors and cost per epoch.
- **Result:** converges in a few epochs and classifies every penguin correctly.

The notebook walks through loading the data, encoding labels, visualizing separability, implementing the model, training, evaluating, and plotting both the decision boundary and the convergence curve.

---

## Limitations

- **Linearly separable data only** — if no straight line can split the classes, it never converges. (This motivates logistic regression next.)
- **No confidence** — the output is a hard $-1$ or $+1$, not a probability.

---

## Files

- `Perceptron.ipynb` — the full implementation and walkthrough.
- `../Datasets/penguins.csv` — the dataset (also available through seaborn).
