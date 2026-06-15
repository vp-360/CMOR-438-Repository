# K-Nearest Neighbors

Every model so far has learned weights from the data. K-Nearest Neighbors does not. It is a **nonparametric** model: there is no training and no parameters. The model just stores the training data, and to classify a new point it finds the closest stored points and goes with the majority. Similar things are near each other, so you are what your neighbors are.

This notebook implements KNN in NumPy and uses it to predict whether a patient has heart disease from their heart measurements.

---

## How It Works

To decide which points are "closest," KNN uses the **Euclidean distance**. This is essentially the same distance formula from geometry, just written in vector form so it works across any number of dimensions at once:

$$d(p, q) = \sqrt{(p - q)^{T}(p - q)} = \sqrt{\sum_i (p_i - q_i)^2}$$

To classify a new point, it measures the distance to every training point, keeps the **k** closest, and takes a majority vote of their labels. There is nothing to train: `fit` only stores the data, and all the work happens at prediction time.

Two choices drive the result. **k** is how many neighbors vote: too small and the model is noisy and chases every stray point, too large and the boundary blurs together. The right k is found by trying a range of values and picking the one with the lowest error. **Scaling** matters because the distance adds up the squared difference of each feature, so a feature on a larger scale would dominate unless every feature is standardized first.

---

## Weighting by Distance

By default every one of the k neighbors gets an equal vote, so the closest point and the k-th closest count the same. A common variation weights each vote by $1/\text{distance}$, so nearer neighbors count more. It can sharpen the prediction and make the choice of k less sensitive, since far neighbors barely contribute. It helps most when the classes have clean local structure; on data with heavy overlap like this one, it usually comes out about even. This notebook uses the default equal-vote version and does not implement distance weighting.

---

## This Implementation

- **Dataset:** the [Cleveland heart disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease) of 303 patients, relabeled so that 1 = heart disease and 0 = healthy.
- **Task:** predict heart disease from two measurements, **thalach** (max heart rate reached) and **oldpeak** (ST depression during exercise).
- **Model:** a `KNN` class where `fit` stores the data and `predict` finds the k nearest by Euclidean distance and takes the majority vote. Features are standardized first.
- **Result:** about 76% test accuracy on the two features, rising to about 83% using all 13.

The notebook loads the data, picks and plots the two features, standardizes and splits them, implements the model, chooses k with an error-vs-k plot, and evaluates with accuracy and a decision-region plot, then repeats using all 13 features.

---

## Limitations

- **Overlapping classes.** When two groups genuinely overlap, with patients who have nearly identical measurements landing in both, KNN cannot separate them. Its mistakes cluster right in that overlap, and no value of k fixes it, because the neighbors there are mixed no matter how many you count.
- **Slow at scale.** There is no model to save, so every prediction compares the new point against the entire training set. That is fine here, but it gets slow as the data grows.

---

## Files

- `K-Nearest-Neighbors.ipynb` — the full implementation and walkthrough.
- `../Datasets/heart.csv` — the dataset.
