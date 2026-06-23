# Principal Component Analysis

This is the second unsupervised algorithm in the repository, and it picks up from the K-Means notebook. There I clustered the seeds dataset using all 7 measurements, but I could only ever plot 2 of them at a time, so I never actually saw the full 7-dimensional shape. PCA is what lets me see it as it takes many features, finds the few directions that hold the most spread, and rebuilds the data along them so I can keep just a couple and still capture almost everything.

PCA never looks at the labels, so it is unsupervised, but it is almost always used as a setup step before a supervised model. This notebook builds it from scratch, uses it to compress the seeds data, and then checks whether that compression actually helps a classifier.

---

## How It Works

First I standardize the data, subtracting each feature's mean and dividing by its standard deviation, so that a large-range feature does not dominate just because of its units. Call that centered and scaled matrix $A$.

PCA then looks for the direction along which the data spreads out the most. That direction is the first principal component, $PC_1$. The next, $PC_2$, is the direction of most remaining spread that is perpendicular to the first, and so on, one component per feature. Each component is a recipe, a weighted mix of all the original features, and those weights are called loading scores.

The clean way to find all of them at once is the singular value decomposition of the centered data,

$$A = U \Sigma V^{T},$$

where the rows of $V^{T}$ are the principal components and the singular values in $\Sigma$ say how much spread each one captures. Squaring a singular value and dividing by the total gives the fraction of variance that component explains, which is what the scree plot graphs. To compress, I project the data onto the first few components and drop the rest.

---

## This Implementation

- **Dataset:** the seeds dataset from the K-Means notebook, 210 wheat kernels with 7 measurements each (area, perimeter, compactness, length, width, asymmetry, groove length) from 3 varieties (Kama, Rosa, Canadian).
- **Task:** compress the 7 features down to 2 so the whole dataset can be plotted at once, then test whether the compressed data still trains a good classifier.
- **Model:** from-scratch PCA through the SVD of the centered data, verified against scikit-learn (the explained-variance ratios match exactly).
- **Result:** the first two components hold about **89%** of the variance, the 2D projection separates the three varieties cleanly along a "size" axis, and compressing 7 features to 2 held a KNN classifier's accuracy and even nudged it up slightly.

The notebook walks through the scree plot to choose how many components to keep, a loadings table and biplot to read what each component is made of, the 2D projection colored by variety, and a before-and-after classifier comparison on the raw versus compressed data.

---

## Limitations

- **PCA only helps when features are redundant.** The seed measurements are all tied to kernel size, so they move together and two components capture almost everything. On data whose features carry independent signal, the variance spreads evenly across many components and compressing throws real information away, so the gain here is not guaranteed elsewhere.
- **The components are not interpretable, and PCA is blind to labels.** $PC_1$ captures "overall size" but is a blend of all 7 features, not a single measurement you can name. And because PCA ranks directions by variance without ever seeing the labels, it can discard a low-variance direction that was actually separating the classes.

---

## Files

- `Principal-Component-Analysis.ipynb` — the full implementation and walkthrough.
- `../Datasets/seeds.csv` — the seeds dataset (210 kernels, 7 features, 3 varieties).
