# Neural Networks

Every model so far has been a **single neuron** which is a weighted sum of the inputs passed through an activation, with the weights learned by gradient descent. A single neuron can only ever cut the space with one straight boundary. A **neural network** stacks neurons into layers, where each neuron draws its own line and the next layer combines them, which lets the network carve out the curved boundaries a single neuron never could. Training it needs **backpropagation**, the chain rule worked backward through the layers to find each layer's error signal, after which it is the same gradient descent as before.

This notebook implements a multilayer network in NumPy and trains it to recognize handwritten digits (MNIST), then runs the same network on clothing images (Fashion-MNIST) to see if it generalizes to a harder problem.

---

## How It Works

**Net input** at each layer $\ell$, a weighted sum of the previous layer's outputs plus a bias:

$$\mathbf{z}^{\ell} = W^{\ell}\mathbf{a}^{\ell-1} + \mathbf{b}^{\ell}$$

**Activation (sigmoid)**, applied elementwise, with $\mathbf{a}^{0} = \mathbf{x}$:

$$\mathbf{a}^{\ell} = \sigma(\mathbf{z}^{\ell}) = \frac{1}{1 + e^{-\mathbf{z}^{\ell}}}$$

**Backpropagation** finds each weight's gradient by computing an error signal $\delta$ for each layer, starting at the output and carrying it backward through the weights:

$$\delta^{L} = (\mathbf{a}^{L} - \mathbf{y})\odot\sigma'(\mathbf{z}^{L}), \qquad \delta^{\ell} = \big((W^{\ell+1})^{T}\delta^{\ell+1}\big)\odot\sigma'(\mathbf{z}^{\ell})$$

**Learning rule** (gradient descent, error signal $\times$ input):

$$W^{\ell} \leftarrow W^{\ell} - \alpha\,\delta^{\ell}(\mathbf{a}^{\ell-1})^{T}, \qquad \mathbf{b}^{\ell} \leftarrow \mathbf{b}^{\ell} - \alpha\,\delta^{\ell}$$

A forward pass sends an image through the layers to produce 10 output scores; backpropagation then sends the error back through the same weights to find each one's share of the blame. The update rule is the same step we have used since the perceptron, now with each layer's error signal in place of $(\hat{y} - y)$.

---

## The Cost Function

$$C = \frac{1}{2}\sum_{k=1}^{10}\big(\hat{y}_k - y_k\big)^2$$

Mean squared error between the 10 output scores and the one-hot label. Minimizing it with gradient descent, where backpropagation supplies the gradient, gives the update rule above.

---

## This Implementation

- **Dataset:** [MNIST](https://en.wikipedia.org/wiki/MNIST_database) (handwritten digits) and [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) (clothing), each 70,000 grayscale 28×28 images across 10 classes, loaded through `keras` or a public mirror.
- **Task:** classify each image into one of 10 categories.
- **Model:** a `DenseNetwork` class with a configurable `layers` list, here `[784, 128, 64, 10]` (two hidden layers), sigmoid activations and Xavier initialization, trained with backpropagation and mini-batch stochastic gradient descent. It tracks MSE per epoch in `errors_`.
- **Result:** around 97–98% test accuracy on MNIST, and around 87% on the harder Fashion-MNIST using the exact same network.

The notebook loads and preprocesses the images (scale, flatten, one-hot encode), implements the network, trains it, plots the cost, and evaluates with accuracy, a confusion matrix, and sample predictions. It also includes an interactive cell to draw your own digit, then runs the whole pipeline again on Fashion-MNIST.

---

## Limitations

- **It ignores the image's structure.** Each image is flattened into a 784-long vector, which throws away the fact that nearby pixels are related. A fully connected network has no notion of shape or locality, part of why it plateaus on the harder Fashion-MNIST. Convolutional networks are built to fix this.
- **It only generalizes to data like its training set.** The network learned centered, plain digits, so inputs that look different (off-center, unusually thick, or stylized) are easily misclassified, as the draw-your-own-digit cell makes clear.

---

## Files

- `Neural-Networks.ipynb` — the full implementation and walkthrough.
- `../Datasets/DATASETS.md` — sources for the MNIST and Fashion-MNIST datasets, which the notebook loads automatically.
