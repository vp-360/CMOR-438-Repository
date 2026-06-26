# Machine Learning Algorithms from Scratch

 This is a collection of core machine learning algorithms implemented in Python, each built to show how it actually works rather than calling a library and moving on. Every algorithm is applied to a real dataset and written up with the underlying math, a worked example, and an honest look at the results and where the method breaks down. Where an algorithm is built from scratch, its output is checked against scikit-learn to confirm it is correct.

The work is split into supervised and unsupervised learning, each with its own folder and overview.

---

## Supervised Learning

Models trained on labeled data. The section builds from a single perceptron up through a from-scratch neural network, then moves to nonparametric methods and ensembles.

- **Perceptron** — from-scratch linear classifier trained by gradient descent; separates two penguin species at 100% accuracy.
- **Linear Regression** — single-neuron model fit by minimizing mean squared error; predicts diabetes progression with an RMSE around 63.
- **Logistic Regression** — from-scratch sigmoid and cross-entropy classifier; diagnoses breast tumors at about 95% accuracy, with a 3D view of the decision surface.
- **Multilayer Perceptron (Neural Network)** — a fully from-scratch feedforward network with backpropagation; 97.4% on MNIST and 88% on Fashion-MNIST, plus a cell to classify a hand-drawn digit.
- **K-Nearest Neighbors (KNN)** — from-scratch distance-based classifier; predicts heart disease at 82.9% using all 13 features.
- **Decision and Regression Trees** — tree models for both classification and regression (scikit-learn), with visualized decision regions and a confusion matrix.
- **Ensemble Methods (Random Forests & Boosting)** — random forests, AdaBoost, and gradient boosting; gradient boosting reaches 82% on wine-quality classification and the best error on auto-MPG regression.

---

## Unsupervised Learning

Finding structure in data that has no labels, either by grouping similar examples or by simplifying the data to its most important pieces.

- **K-Means Clustering** — from-scratch clustering that recovers 3 wheat varieties from 7 kernel measurements, matching the true varieties at an adjusted Rand index of 0.77 (identical to scikit-learn).
- **Principal Component Analysis** — from-scratch PCA through the singular value decomposition; compresses 7 features into 2 while keeping about 89% of the variance, and shows the compression preserves a classifier's accuracy.

---

## Repository Structure

```
├── Supervised Learning/
│   ├── Datasets/
│   ├── 1) Perceptron/
│   ├── 2) Linear Regression/
│   ├── 3) Logistic Regression/
│   ├── 4) Multilayer Perceptron (Neural Network)/
│   ├── 5) K-Nearest Neighbors (KNN)/
│   ├── 6) Decision and Regression Trees/
│   └── 7) Ensemble Methods (Random Forests & Boosting)/
└── Unsupervised Learning/
    ├── Datasets/
    ├── 1) K-Means Clustering/
    └── 2) Principal Component Analysis/
```

Each algorithm folder holds a Jupyter notebook and a README that explains that algorithm in detail, with shared datasets kept in each section's `Datasets/` folder.

---

## Tech Stack

Python · NumPy · pandas · matplotlib · seaborn · scikit-learn

---

*Developed for CMOR 438 (Data Science and Machine Learning) at Rice University.*
