# Decision and Regression Trees

A **decision tree** classifies by asking a series of yes/no questions about the features ("is glucose less than 120? go left, else go right"), repeated until it reaches a leaf that gives the answer. Geometrically it chops the feature space into **rectangular blocks** with straight, axis-aligned lines, a different shape from the curves and bumpy regions of the earlier models. The same idea does **regression**: each leaf predicts the *average* value of the points that land in it, so the prediction comes out as a flat staircase. The big appeal is that a trained tree is **easy to read**, you can follow it top to bottom and see exactly why it made a prediction.

This notebook builds both kinds with scikit-learn: a decision tree to predict diabetes, and a regression tree to predict medical charges.

---

## How It Works

To classify, the tree splits the data one feature at a time, choosing the cutoff that makes the resulting groups as **pure** as possible. A group is pure when it contains only one class, every point is diabetic or none of them are, for example. Purity is measured by the **Gini index**:

$$\text{Gini} = 1 - \sum_k p_k^2$$

**Low Gini = pure.** If a group is all one class, $p = 1$ for that class and 0 for everything else, so Gini $= 1 - 1^2 = 0$. If it is a perfect 50/50 mix, Gini $= 1 - (0.5^2 + 0.5^2) = 0.5$, the worst possible. The tree always wants to drive Gini toward zero.

When picking a split, the tree doesn't just look at one child, it computes a **weighted average** of both children's Gini scores, weighted by how many points each child gets. This means size matters: a child with 900 points counts for far more than a child with 1 point. So even if one side is perfectly pure, a messy 900-point side still makes the overall split bad. The best split is the one that drops the weighted Gini the most compared to the parent.

The process repeats until the tree hits its `max_depth` or the leaves are already pure. To predict, a new point just follows the splits down to a leaf and takes that leaf's majority class.

A **regression tree** works the same way, except each leaf predicts the **average** target value of its points, and splits are chosen to make those values as close together as possible (lowest error) instead of purest. That is why its prediction is a staircase, flat within each region.

There is no training in the gradient-descent sense and no feature scaling needed, since the tree only ever compares one feature against a cutoff at a time.

---

## This Implementation

- **Classification — Pima diabetes:** predict diabetes (yes/no) from **glucose** and **BMI** with a `DecisionTreeClassifier`. The notebook draws the tree with `plot_tree`, shows the rectangular decision regions on both the training and test data, and evaluates with accuracy, a confusion matrix, and a classification report (about **76%** test accuracy).
- **Regression — insurance:** predict **medical charges** with a `DecisionTreeRegressor`, first on a single feature (**age**) to show the staircase, then on all the features, reporting the mean absolute error as the depth changes.
- **Overfitting:** both halves show what happens when the tree grows too deep, the regions break into tiny islands and the error starts climbing again, the tree memorizing the data instead of learning the pattern.

---

## Limitations

- **Overfitting.** This is the tree's biggest weakness, and the notebook shows it directly: let the depth grow and it carves the space into little islands around single points, memorizing noise instead of the real pattern. Capping the depth keeps it in check, but a single tree is always fighting this.
- **Axis-aligned splits only.** Every split is "one feature above or below a cutoff," so the boundaries are always built from straight rectangular blocks. The tree can't draw a diagonal, and rotating the data can produce a completely different, worse tree.

---

## Files

- `Decision-and-Regression-Trees.ipynb` — the full implementation and walkthrough.
- `../Datasets/Diabetes Dataset for Decision Trees.csv` — the Pima diabetes data (classification).
- `../Datasets/insurance.csv` — the medical insurance charges data (regression).
