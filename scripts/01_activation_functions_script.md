# Video Script: Activation Functions for Deep Learning

**Notebook:** `final_activation_functions_tutorial.ipynb`  
**Author:** Nitish Chowdary Ratakonda  
**Estimated read time:** ~15–18 minutes

> Read this script aloud while scrolling through the executed notebook. Each section maps one-to-one to a cell.

---

## Intro

Hey everyone, I'm Nitish. In this video I'm walking through my executed Colab on **activation functions for deep learning**. Activation functions are the components that give neural networks the ability to learn non-linear patterns — without them, a deep network collapses into a single linear model. I'll go cell by cell, explain what each one does, and point out what the output is telling us.

---

## Cell 0 — Colab badge (markdown)

This first cell is just the "Open in Colab" badge. It's an HTML anchor that lets anyone click straight from the GitHub copy into a live Colab session. Nothing to execute here.

## Cell 1 — Title and Table of Contents (markdown)

This is the title slide: *"Activation Functions for Deep Learning — From Zero to Hero."* The table of contents lists the eight parts we'll cover — from why activations matter, to classic ones like sigmoid and tanh, the ReLU revolution, ReLU variants, modern activations used in transformers, output-layer activations, a practical selection guide, and finally a hands-on training comparison.

## Cell 2 — Setup and imports (code)

This cell installs and imports everything we need: `numpy`, `matplotlib`, `torch`, and `sklearn`. We also fix the random seeds to 42 for reproducibility and set a clean Seaborn plotting style. The output just confirms "Libraries loaded successfully!"

## Cell 3 — Why linear networks fail (markdown)

This sets up Part I. The key mathematical point: if you stack two linear layers `W1·x + b1` and `W2·h + b2` without any activation in between, the whole thing algebraically collapses into one linear transformation `W'·x + b'`. No matter how many layers you add, without non-linearity you still only have a linear model — which can only separate data with straight lines.

## Cell 4 — Visualizing non-linear problems (code)

Here we plot three classic datasets that a linear model cannot solve: the XOR problem with four points you can't separate with any single straight line, the two-moons dataset that needs a curved boundary, and the concentric-circles dataset that needs a circular boundary. The output shows three scatter plots and a printed message: *"These problems cannot be solved with linear models — activation functions add the non-linearity needed."*

## Cell 5 — What activations do (markdown)

A concise definition: an activation function σ is applied element-wise after the linear transformation, so `a = σ(Wx + b)`. It lists the five properties we want: non-linearity, differentiability, non-saturation, zero-centering, and computational efficiency.

## Cell 6 — Classic activations: Sigmoid and Tanh (markdown)

This introduces sigmoid, with output range zero to one, and tanh, with range negative one to one. It lists why sigmoid was historically important, but also its problems: vanishing gradients at the extremes, non-zero-centered outputs, and expensive exponential computation. Tanh fixes zero-centering but still saturates.

## Cell 7 — Plotting sigmoid and tanh and their derivatives (code)

We define sigmoid, tanh, and their derivatives as plain NumPy functions, then plot a two-by-two grid. Top-left is sigmoid — a smooth S-curve squashed between zero and one. Top-right is its derivative, maxing out at only 0.25, with red bands highlighting the saturated regions where the gradient is essentially zero. Bottom-left is tanh — also S-shaped but centered at zero. Bottom-right is its derivative, maxing at one but still saturating at the extremes. The printed output drives home the **vanishing gradient problem**: because sigmoid's max derivative is 0.25, stacking four layers multiplies the gradient by 0.25 to the fourth power — about 0.004. Deep networks built on sigmoid simply can't train.

## Cell 8 — The ReLU revolution (markdown)

This cell introduces ReLU, defined as `max(0, x)`, with derivative 1 for positive inputs and 0 otherwise. It explains why ReLU changed everything in the 2010s: no vanishing gradient on the positive side, sparse activations, it's computationally cheap, and it's biologically plausible. But there's a catch — the **dying ReLU problem**: if a neuron's input stays negative, its output and gradient are both zero, and it never recovers.

## Cell 9 — Visualizing ReLU and dying ReLU (code)

Three-panel figure. Left: ReLU itself — flat at zero for negative x, linear for positive x. Middle: the derivative — a step function jumping from zero to one at the origin. Right: a histogram simulation of what happens when a neuron's inputs are biased negative — the blue histogram shows negative inputs, and the red histogram shows all outputs collapsed to zero. That's a dead neuron. The printout emphasizes that ReLU made deep networks like AlexNet possible in 2012 and is still the default for hidden layers.

## Cell 10 — ReLU variants (markdown)

This introduces four fixes for dying ReLU: **Leaky ReLU** (small slope α for negative inputs, typically 0.01), **PReLU** (same idea but α is learned), **ELU** (exponential curve for negatives, smooth at zero), and **SELU** (scaled ELU with specific constants that make activations self-normalize to zero mean and unit variance).

## Cell 11 — Plotting ReLU variants (code)

Two-by-two figure. Top-left compares all four variants overlaid — you can see Leaky ReLU has a gentle slope on the negative side, ELU curves smoothly, and SELU is similar but scaled. Top-right shows Leaky ReLU with alpha values 0.01, 0.1, and 0.3 — higher alpha means a steeper negative slope. Bottom-left zooms in near zero to contrast ReLU's sharp corner with ELU's smooth transition. Bottom-right is a color-coded comparison table showing which variants suffer from dead neurons, which are zero-centered, which are smooth, and their relative speeds.

## Cell 12 — Modern activations (markdown)

Introduces the state-of-the-art: **GELU**, used in BERT, GPT, and most transformers — a smooth Gaussian-weighted version of ReLU. **Swish or SiLU**, discovered by Google through neural architecture search, defined as `x · sigmoid(x)` — self-gated and non-monotonic. And **Mish**, used in YOLOv4, even smoother than Swish.

## Cell 13 — Plotting modern activations (code)

Three-panel figure. Left: GELU, Swish, and Mish all overlaid against a dashed ReLU reference — they're all smooth approximations of ReLU. Middle: a zoomed view of the interesting region near zero, where these functions dip slightly negative before coming back up — that non-monotonicity is one reason they work so well. Right: a table listing which famous models use which activation — ReLU in ResNet/VGG/AlexNet, GELU in BERT/GPT/ViT/LLaMA, Swish in EfficientNet and Stable Diffusion, Mish in YOLO. The takeaway: GELU now dominates large language models.

## Cell 14 — Output-layer activations (markdown)

For the *output* layer, the activation depends on the task. **Sigmoid** for binary classification (paired with binary cross-entropy loss). **Softmax** for multi-class classification (paired with cross-entropy). **No activation** (linear) for regression (paired with MSE).

## Cell 15 — Plotting output activations (code)

Three panels. Left: a bar chart showing five logits passed through sigmoid — values below 0.5 classify as the negative class in red, above 0.5 in blue. Middle: a softmax example with four class logits being normalized to probabilities that sum to 1.0 — note that the highest logit dominates but doesn't completely take over. Right: a reference table mapping classification task types to their recommended output activation and loss function.

## Cell 16 — Practical activation-selection guide (markdown)

An ASCII decision tree for choosing hidden-layer activations: transformers go with GELU, CNNs with ReLU or SiLU, MLPs with ReLU or SELU. Below that is a detailed table listing recommended, alternative, and avoid-at-all-costs activations for transformers, CNNs, RNNs, MLPs, and GANs.

## Cell 17 — Setup for hands-on comparison (markdown)

This is a short header saying we're now going to actually train networks with different activations and see who wins.

## Cell 18 — Training with different activations (code)

We generate the two-moons dataset with a thousand samples, then define a flexible three-layer network whose activation function is configurable. We train it seven times — once each with ReLU, Leaky ReLU, ELU, GELU, SiLU, tanh, and sigmoid — for one hundred epochs with Adam. The printed output shows the final accuracy for each. Modern activations like GELU, SiLU, and ELU typically hit the high nineties, while sigmoid lags noticeably because it saturates too easily on this network depth.

## Cell 19 — Visualizing the comparison (code)

A two-by-four grid of plots. The first plot overlays all seven loss curves — you can literally see sigmoid plateau early while the modern activations keep driving the loss down. The next six panels show the decision boundaries each activation learned on the moons dataset — ReLU, GELU, and friends draw a clean curve between the two moons, while sigmoid's boundary is noticeably worse. The final panel is a bar chart of final accuracies — visual proof that activation choice matters.

## Cell 20 — Summary (markdown)

The wrap-up cell. Six key takeaways: without activations, networks are just linear; classic activations suffer from vanishing gradients; ReLU enabled deep networks; ReLU variants solve dying neurons; modern smooth activations like GELU dominate transformers; and output activations depend on your task. It also includes a quick-reference block and a timeline of activation evolution from the 1940s perceptron to today's transformer-dominating GELU.

---

## Outro

That's the full activation-functions tutorial. The big picture: activation choice is one of those small decisions that quietly determines whether your network trains at all. For modern hidden layers, default to ReLU or GELU; for outputs, pick sigmoid, softmax, or linear based on your task. Thanks for watching — check the description for the notebook and the rest of the playlist.
