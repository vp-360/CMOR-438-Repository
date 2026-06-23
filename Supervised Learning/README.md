# Supervised Learning

Supervised learning is the branch of machine learning where the data comes with answers. Every training example is a pair, some input features and the correct output label. The algorithm's job is to learn the mapping from inputs to outputs by adjusting its internal settings until its predictions are as close to the real labels as it can get, so that when a new, unlabeled example comes in, it can predict the answer.

This section works through seven supervised algorithms. The order is not random, it follows how the ideas build on each other, and each folder has its own README that goes deeper with the math and a worked example.

---

## How It Works

Every supervised algorithm relies on **labeled data**, where each training example carries the correct answer, which is what separates supervised from unsupervised learning. Most of these notebooks follow the same basic pipeline:

1. **Collect labeled data** — gather a dataset where every example has both its features and the correct label.
2. **Prepare the data** — clean it, scale the features when the algorithm needs it, and split it into training and testing sets.
3. **Train the model** — feed it the training data and let it adjust its parameters to reduce **a loss function**, a measure of how wrong the model is (mean squared error for regression, cross-entropy for classification), which is the thing training tries to minimize.
4. **Evaluate** — measure how well it does on the held-out test set using **a train/test split**, because a good training score alone is not enough. Accuracy, RMSE, and a confusion matrix give concrete numbers to compare models against each other.
5. **Predict** — once it generalizes well, use it to label new, unseen examples.

---

## How These Algorithms Fit Together

The first four are really similar because at the center is a single neuron: it takes a weighted sum of the inputs, passes it through an activation function, and learns its weights by gradient descent on a cost function. By changing the activation and the cost and you move from one algorithm to the next. The perceptron uses a hard threshold to draw a straight boundary. Linear regression drops the activation entirely and minimizes squared error to predict a number. Logistic regression swaps in a sigmoid and cross-entropy to predict a probability. Neural networks then stack many of these neurons into layers and train them together with backpropagation.

The next two step away from that idea completely. K-nearest neighbors and decision trees are nonparametric, meaning there are no weights to learn. They make predictions straight from the shape of the data, by looking at nearby points or by splitting the feature space into regions.

The last one combines models instead of building a single better one. Ensemble methods take many trees and pool them, through bagging, boosting, or gradient boosting, into one predictor that is stronger than any tree alone.

---

## The Algorithms

1. **The Perceptron** — the original single-neuron classifier; a threshold on a weighted sum that learns a linear boundary. Built from scratch on the Palmer penguins (Adelie vs Gentoo), reaching 100% on two clean features.

2. **Linear Regression** — the same neuron with no activation, fit by minimizing mean squared error to predict a continuous value. Predicts diabetes progression from BMI, with an RMSE around 63.

3. **Logistic Regression** — a sigmoid activation and cross-entropy loss turn the neuron into a probability-based classifier. Diagnoses breast tumors, about 95% accuracy on two features, with a 3D look at the decision surface.

4. **Multilayer Perceptron (Neural Network)** — many neurons stacked into layers and trained by backpropagation. A from-scratch network on MNIST and Fashion-MNIST, hitting 97.4% and 88%, with a cell to draw your own digit.

5. **K-Nearest Neighbors (KNN)** — the first nonparametric method; classify a point by the majority vote of its closest neighbors, with no training step at all. Predicts heart disease, 82.9% using all 13 features.

6. **Decision and Regression Trees** — split the data with yes/no questions into regions, which works for both classification and regression. Pima diabetes (about 76%) and a regression tree on medical insurance charges.

7. **Ensemble Methods (Random Forests & Boosting)** — combine many trees into one stronger model through random forests (bagging), AdaBoost, and gradient boosting. Red wine quality for classification (gradient boosting reaching 82%) and auto MPG for regression.

---

## Structure

Each algorithm lives in its own numbered folder with a notebook and a README that explains it in detail. All the datasets are kept in `Datasets/` folder.
