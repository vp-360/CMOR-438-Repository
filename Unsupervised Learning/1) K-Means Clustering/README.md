# K-Means Clustering

This is the first unsupervised algorithm in the repository, so there are no labels to learn from. Instead of predicting a known answer, the goal is to find the natural groups hiding in the data on their own. K-Means is the simplest way to do that: you guess how many groups there are (**k**), then let the algorithm sort the points into k clusters by their similarity.

This notebook builds K-Means from scratch and uses it to group seeds into varieties from their measurements alone.

---

## How It Works

You start by guessing **k** and dropping k cluster centers (centroids) at random. Then you repeat two steps until nothing changes:

1. **Assign** every point to its nearest centroid, using straight-line (Euclidean) distance, $\sqrt{\sum_j (x_j - c_j)^2}$, summed over all the features.
2. **Update** each centroid to the **mean** of the points that joined it (that average is the "Means" in K-Means).

Each round the centroids drift toward the middle of their groups, and you stop once they stop moving, which is convergence. Behind the scenes this is always lowering the **inertia**, the total squared distance from every point to its centroid, and both steps can only shrink it, which is why it always settles.

Two practical notes. K-Means can't tell you how many groups there are, so you pick k with the **elbow method**: plot the inertia for a range of k and look for the bend where adding clusters stops helping. And the whole thing works in any number of dimensions, the distance just sums over more features, so the only thing that changes with more features is that you can no longer plot it directly.

---

## This Implementation

- **Dataset:** the seeds dataset, 210 wheat kernels described by 7 measurements (area, perimeter, compactness, length, width, asymmetry, groove length) from 3 varieties (Kama, Rosa, Canadian).
- **Task:** cluster the kernels with the variety labels hidden, then reveal them at the end to check the result.
- **Model:** from-scratch K-Means on all 7 standardized features with k = 3, keeping the tightest of several random starts.
- **Result:** the clusters recover the varieties well, an adjusted Rand index of about **0.77**, matching scikit-learn exactly, and the elbow plot independently points at k = 3.

The notebook shows convergence as an inertia-vs-iteration curve (since 7 dimensions can't be plotted), then drops to 2 features to draw the centroids drifting step by step, views the clusters next to the true varieties, and validates with a crosstab and the ARI.

---

## Limitations

- **You pick k, and the starting point matters.** K-Means can't find the number of clusters on its own (the elbow only suggests it), and because it begins from random centroids, different starts can land in different places, so I ran it several times and kept the tightest.
- **It assumes round, similar-sized clusters.** Since it judges everything by distance to a center, it struggles with stretched or oddly shaped groups, and it forces every point into a cluster, so it has no way to flag an outlier. Those two gaps are what the next algorithm, DBSCAN, is built to handle.

---

## Files

- `K-Means-Clustering.ipynb` — the full implementation and walkthrough.
- `../Datasets/seeds.csv` — the seeds dataset (210 kernels, 7 features, 3 varieties).
