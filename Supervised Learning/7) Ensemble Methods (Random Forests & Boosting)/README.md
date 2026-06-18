# Ensemble Methods (Random Forests & Boosting)

**Ensemble methods** combine many decision trees into one stronger model. A single tree tends to overfit, memorizing noise in the training data, but a crowd of trees, each a little wrong in its own way, averages out to something far more reliable. There are two main strategies: **bagging** (train many trees independently and let them vote) and **boosting** (train trees one at a time, each fixing the mistakes of the last). This notebook covers both, using random forests for bagging and AdaBoost and gradient boosting for the boosting side.

---

## How It Works

### Random Forests (Bagging)

A random forest trains many decision trees in parallel and combines their predictions. Two sources of randomness make the trees different from each other:

1. **Bootstrap sampling:** each tree trains on a random sample of the rows, drawn with replacement.
2. **Random feature subsets:** at each split, a tree only considers a random subset of the features (typically $\sqrt{p}$).

For classification the trees vote and the majority wins; for regression the predictions are averaged. Because each tree overfits in its own random direction, combining them cancels out the noise. This "bootstrap + aggregate" combination is called **bagging**.

### AdaBoost

AdaBoost builds a chain of **weak learners** (depth-1 decision stumps). After each stump, the algorithm **reweights** the training data so that misclassified points get more attention in the next round. The final prediction is a weighted vote where more accurate stumps get a louder say. For regression, it reweights the points with the largest errors instead, though this makes it sensitive to outliers.

### Gradient Boosting

Gradient boosting also chains trees sequentially, but each new tree fits the **residual errors** (the gap between the current prediction and the truth) rather than reweighting points. The prediction is the sum of all the trees, and each one nudges the total closer to the correct answer. This is the same **gradient descent** idea from neural networks: instead of adjusting weights, we add a tree, and the residual error plays the role of the gradient.

Three knobs control gradient boosting: **`n_estimators`** (how many trees), **`learning_rate`** (how big each correction is), and **`max_depth`** (how deep each tree grows). A smaller learning rate needs more trees but overfits less.

---

## This Implementation

- **Classification — red wine quality:** predict whether a wine is good (quality $\geq$ 6) from 11 chemical features. Compares a single tree, random forest (200 trees), AdaBoost (1000 depth-1 stumps), and gradient boosting (200 trees, depth 2, lr=0.3). Decision boundary plots on a two-feature subset (alcohol and sulphates) show how each model carves up the space differently.
- **Regression — auto MPG:** predict a car's fuel efficiency from its specs. Compares the same four models using MAE. A three-tree residual demo on the weight feature shows gradient boosting's mechanics step by step.
- **Feature importance:** the random forest's impurity-based feature importances show which measurements the model relied on most (alcohol dominates for wine quality).
- **Hyperparameter tuning:** grid searches over `n_estimators`, `learning_rate`, and `max_depth` for each ensemble method, with results sorted so the best combination is always the last line.
- **n_estimators curves:** plots showing how the error changes as trees are added, for both the random forest and gradient boosting, with the single tree as a baseline.

---

## Results

| Model | Wine Accuracy | MPG MAE |
|---|---|---|
| Single tree | 74.0% | 2.05 mpg |
| Random forest | 80.2% | 1.70 mpg |
| AdaBoost | 76.8% | 3.38 mpg |
| Gradient boosting | 82.0% | 1.67 mpg |

AdaBoost regression performs worse than the single tree because its reweighting mechanism over-focuses on outliers, confirming that it is primarily a classification tool and gradient boosting is the stronger choice for regression.

---

## Limitations

- **Interpretability.** A single tree can be read top to bottom, but an ensemble of hundreds of trees is a black box. Feature importance gives some insight, but you can't trace exactly why a specific prediction was made.
- **AdaBoost regression weakness.** AdaBoost's reweighting of large errors makes it sensitive to outliers in regression tasks, often performing worse than a well-tuned single tree.
- **Hyperparameter sensitivity.** Gradient boosting's performance depends heavily on the combination of `n_estimators`, `learning_rate`, and `max_depth`. A grid search helps, but the best settings for one data split may not hold for another.

---

## Files

- `Ensemble Methods (Random Forests & Boosting).ipynb` — the full implementation and walkthrough.
- `../Datasets/winequality.csv` — the red wine quality data (classification).
- `../Datasets/auto-mpg.csv` — the auto MPG data (regression).
