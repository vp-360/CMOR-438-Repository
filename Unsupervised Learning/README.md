# Unsupervised Learning

Unsupervised learning is the half of machine learning where the data has no labels. There are no correct answers to train against, so instead of learning to predict a known output, the algorithm has to find the structure already sitting in the features on its own, whether that means grouping similar examples together or simplifying the data down to its most important pieces.

This section works through that idea in two directions: clustering, which groups the rows, and dimensionality reduction, which simplifies the columns. Each folder has its own README that goes deeper with the math and a worked example.

---

## How Unsupervised Learning Works

The pipeline is similar to supervised learning but without a label to aim at:

1. **Collect unlabeled data** — gather a dataset of features with no target column.
2. **Prepare the data** — clean it and standardize the features, which matters a lot here because these methods judge everything by distance or spread.
3. **Find the structure** — let the algorithm group similar points (clustering) or find the directions that carry the most information (dimensionality reduction).
4. **Check the result** — without labels there is no accuracy score, so I use datasets that happen to come with labels and compare the discovered structure against them, even though the algorithm never saw them.
5. **Use it** — visualize the result, compress the data, or hand it to a supervised model as a cleaner input.

---

## Key Characteristics

- **No labels** — the algorithm works only from the features, with no correct answer to copy.
- **Structure over prediction** — the goal is to group or simplify the data, not to predict a specific output.
- **Scaling matters** — since these methods rely on distance and spread, the features are standardized first so one large-range feature does not dominate.
- **Harder to validate** — there is no built-in accuracy, so I lean on datasets with known labels (checking with the adjusted Rand index) or on measures like explained variance.

---

## How These Algorithms Fit Together

The section moves in two directions. The first is clustering, grouping rows that are similar. K-Means does this by distance: it drops a few centers, assigns each point to the nearest one, and repeats, which works well when the groups are roughly round. The second is dimensionality reduction, simplifying the columns. PCA finds the handful of directions along which the data varies the most and rebuilds it along them.

The two connect directly. I clustered the seeds in seven dimensions with K-Means but could only ever plot two features at a time, so I never saw the full shape. PCA is what finally let me see all seven at once in a single 2D plot, which is exactly the picture the clustering was working with.

---

## The Algorithms

1. **K-Means Clustering** — drops k centers, assigns each point to the nearest one, then moves the centers to the mean of their points and repeats until they settle. Clusters the seeds dataset (210 wheat kernels, 7 features) into 3 varieties with the labels hidden, recovering them at an adjusted Rand index of about 0.77, matching scikit-learn.

2. **Principal Component Analysis** — finds the directions of most spread through the SVD of the data, then projects onto the top few to compress many features into a couple. Compresses the same seeds data from 7 features to 2 (holding about 89% of the variance) and shows the compression keeps a classifier's accuracy.

---

## Structure

Each algorithm lives in its own numbered folder with a notebook and a README that explains it in detail. Shared datasets are kept in `Datasets/` so multiple notebooks can reference the same file (both notebooks here use the same `seeds.csv`).
