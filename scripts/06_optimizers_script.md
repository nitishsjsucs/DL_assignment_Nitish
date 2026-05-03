# Video Script: Optimizers for Deep Learning

**Notebook:** `final_optimizers_deep_learning_tutorial.ipynb`  
**Author:** Nitish Chowdary Ratakonda  
**Estimated read time:** ~25–30 minutes

> Read this script aloud while scrolling through the executed notebook. Each section maps one-to-one to a cell.

---

## Intro

Hey everyone, I'm Nitish. In this video I'm walking through my executed Colab on **optimizers for deep learning**. The optimizer is the engine of learning — it's what actually updates your model's weights. I'll go from basic gradient descent all the way through Adam, AdamW, and cutting-edge optimizers like SAM and Lion, with visualizations of loss landscapes and optimizer paths at every step. Let's go.

---

## Cell 0 — Colab badge (markdown)

The Colab badge for live editing.

## Cell 1 — Title and Table of Contents (markdown)

Title: *"Optimizers for Deep Learning — From Zero to Hero."* The table of contents lists ten parts: introduction, gradient descent basics, problems with vanilla GD, stochastic gradient descent, momentum methods, adaptive learning rates, the Adam family, learning-rate schedules, practical comparisons, and modern advanced optimizers.

## Cell 2 — Install dependencies (code)

Pip installs NumPy, Matplotlib, PyTorch, torchvision, and scikit-learn. No output to speak of.

## Cell 3 — Imports and plot style (code)

Imports everything we need — NumPy, Matplotlib with 3D plotting, PyTorch, DataLoader, sklearn's make_moons. Sets random seeds and a clean Seaborn plot style. Prints "All libraries loaded successfully!"

## Cell 4 — Why optimizers matter (markdown)

Frames the central problem: deep learning is about finding the best parameters that minimize a loss. The optimizer decides *how* to update those parameters given gradient information. Without a good optimizer, no architecture trains well.

## Cell 5 — Optimization journey visualization (code)

Three panels. Left: a simple 1D loss landscape `x² + 0.5·sin(5x)`, with a marker showing the starting point and an arrow pointing toward the minimum. Middle: a typical training curve showing training and validation loss dropping over 50 epochs — that's the curve an optimizer actually produces. Right: a block diagram showing the standard training loop — data into the model, model output into the loss, loss backprop gives gradients, and the *optimizer* uses those gradients to update weights.

## Cell 6 — What is a loss landscape (markdown)

Explains loss landscapes using the mountain-hiker analogy. Each point represents a set of parameters, the height is the loss. Our goal: find the lowest valley. In real networks there are millions of parameters, so the landscape is millions of dimensions — impossible to visualize directly, but we can build intuition in 2D.

## Cell 7 — 3D loss landscapes (code)

Four 3D surface plots. Top-left: a simple bowl — `x² + y²` — easy to optimize. Top-right: an elongated ravine — `x² + 10y²` — the loss drops fast in one direction and slowly in the other, causing oscillation problems. Bottom-left: a saddle point — `x² − y²` — a minimum in X but a maximum in Y. Bottom-right: a multi-modal landscape with multiple minima. Real neural network loss landscapes combine all these features.

## Cell 8 — Contour plots intro (markdown)

Introduces contour plots — the bird's-eye view of a loss landscape, like a topographic map. Closer contour lines mean a steeper slope. Much easier to visualize optimization paths on contour plots than on 3D surfaces.

## Cell 9 — Contour plots for each landscape (code)

The same four landscapes as contour plots. You can now clearly see the concentric rings of the simple bowl, the stretched ellipses of the ravine, the X-shape of the saddle point, and the multiple separated minima of the multimodal landscape.

## Cell 10 — Why deep learning optimization is hard (markdown)

Lists the six core challenges of optimizing real neural networks: high dimensionality (millions of parameters), non-convexity (many local minima), saddle points (flat regions with zero gradient), ill-conditioning (curvatures differ wildly across directions), noise from mini-batch gradients, and vanishing or exploding gradients. Modern optimizers are specifically designed to handle these.

## Cell 11 — Visualizing the three challenges (code)

Three panels. Left: a saddle-point contour plot with arrows showing that the gradient pushes you toward the saddle in one direction and *away* from it in the other. Middle: an ill-conditioned ravine with a simulated gradient-descent path zig-zagging oscillations in the steep direction. Right: a multimodal landscape with a global minimum starred in red and several local minima in orange — showing that a naive optimizer can get stuck in the wrong valley.

## Cell 12 — What is a gradient (markdown)

Defines the gradient: a vector pointing in the direction of steepest *increase*. To minimize loss, we go in the opposite direction — the negative gradient — which gives us the steepest descent.

## Cell 13 — Visualizing gradients (code)

Two plots. Left: the 1D parabola `L(w) = w²`, with five points marked and tangent lines drawn at each — the slope of each tangent is the gradient at that point. Right: a 2D gradient field on the `x² + y²` bowl — red arrows pointing uphill toward the rim and blue arrows pointing downhill toward the center. That's exactly the direction we want to move during optimization.

## Cell 14 — Vanilla gradient descent (markdown)

States the basic gradient-descent update: `W_new = W_old − η · ∇L(W_old)`, where η is the learning rate. The algorithm in three steps: initialize parameters randomly, compute the gradient, take a step in the negative gradient direction — repeat until converged.

## Cell 15 — Vanilla GD implementation (code)

We implement `VanillaGradientDescent` as a small Python class that tracks its own trajectory. We apply it to the simple bowl `L = w1² + w2²` starting from (2, 2) with learning rate 0.2, run 20 steps, and print the starting point, final point, and starting and final loss values. The final loss is essentially zero — converged to the minimum.

## Cell 16 — Visualizing GD path (code)

Two plots. Left: the bowl contour with red dots connecting the starting point through 20 steps down to the yellow star at the minimum — the classic gradient-descent trajectory. Right: loss versus iteration on a log scale, dropping smoothly toward zero.

## Cell 17 — The learning rate (markdown)

The learning rate is the single most important hyperparameter. Too small: forever to converge. Just right: efficient. Too large: overshoots and may diverge. The table summarizes each case.

## Cell 18 — Learning-rate sweep (code)

A 2-by-3 grid showing gradient-descent paths on the bowl for six different learning rates: 0.01 (painfully slow), 0.1 (good), 0.5 (fast), 0.9 (risky), 1.0 (critical — barely converges), and 1.1 (diverges completely). You can visually see each behavior. Each panel is color-coded by final loss — green for converged, orange for struggling, red for diverged. This single cell drives home why learning rate matters so much.

## Cell 19 — Ravine problem (markdown)

Explains ill-conditioning. In real networks, loss landscapes have ravines — steep in some directions, shallow in others. Gradient descent oscillates in the steep direction and makes slow progress in the shallow direction. This causes the classic zig-zag pattern.

## Cell 20 — Visualizing the ravine problem (code)

Three panels, showing the same ravine with three "condition numbers" — 1 (well-conditioned), 10 (moderate), 100 (severe). For each one, we run vanilla GD and plot the path. As conditioning worsens, the path zig-zags more violently. Each panel shows the oscillation count. The printed takeaway: higher condition numbers cause more oscillation.

## Cell 21 — Saddle points (markdown)

The hidden danger in high-dimensional optimization. A saddle point has zero gradient but isn't a minimum — it's a minimum in some directions and a maximum in others. The gradient gets tiny near a saddle, so optimization slows to a crawl. Research from Dauphin et al. 2014 showed that in high dimensions saddle points are *more* common than local minima.

## Cell 22 — Saddle point visualization (code)

Three panels. Left: the 3D saddle surface — literally shaped like a horse saddle, with the saddle point marked. Middle: the 2D contour with three gradient-descent runs from different starting points — one trapped near the saddle, one that escapes, one diagonal. Right: a plot of gradient magnitude, showing that it drops to nearly zero at the saddle point. That's why optimization stalls there.

## Cell 23 — Stochastic gradient descent (markdown)

Introduces SGD. Instead of computing the gradient on the full dataset (batch GD), we compute it on a small mini-batch — or even a single sample. This makes updates noisy but dramatically cheaper. The key insight: **noise is a feature, not a bug** — it helps escape shallow local minima and provides implicit regularization. In practice, mini-batches of 32 to 256 samples balance noise and stability.

## Cell 24 — Batch GD vs SGD (code)

Three panels comparing batch GD to SGD on a multimodal landscape. Left: batch GD from a given starting point gets stuck in a local minimum. Middle: SGD with the same start, adding noise to each gradient, wanders more chaotically but can jump out of the local minimum and find a better one. Right: the loss curves for both — batch GD is smooth but plateaus at a suboptimal value, SGD is noisier but reaches lower loss.

## Cell 25 — Batch-size effects (markdown)

Quick reference on batch-size trade-offs. Small batches: more noise, more frequent updates, less memory. Large batches: less noise, more stable, better GPU utilization. Typical sizes: 32-256 for vision, 8-64 for NLP transformers, 1000+ for distributed training. The linear scaling rule: when you double the batch size, double the learning rate.

## Cell 26 — Momentum intro (markdown)

Introduces momentum with the physics analogy — a ball rolling down a hill accelerates, carries momentum through small bumps, and can roll over small obstacles. The math: we track a running "velocity" `v = β·v + ∇L` and update `W = W − η·v`, typically with momentum coefficient β around 0.9. Benefits: dampens oscillations in ravines, accelerates convergence in consistent directions, and helps escape saddle points.

## Cell 27 — GD vs Momentum (code)

Three panels on a severe ravine landscape. Left: vanilla GD oscillates noticeably. Middle: SGD with momentum takes a much smoother path — the oscillations in the steep direction cancel out over time while the shallow-direction motion accumulates. Right: loss curves showing momentum converging significantly faster.

## Cell 28 — Nesterov accelerated gradient (markdown)

Introduces NAG — a smart tweak over regular momentum. Regular momentum: compute gradient at current position, then move. Nesterov: first *look ahead* to where the momentum would take you, compute the gradient *there*, then move. This anticipatory update corrects overshooting faster and is provably faster for convex problems.

## Cell 29 — Adaptive learning rates intro (markdown)

Explains why we need adaptive methods. Different parameters may need different learning rates — sparse features need bigger steps, frequent features need smaller ones. Instead of a single global learning rate, adaptive methods automatically scale the learning rate per parameter based on gradient history. First adaptive method: AdaGrad, which accumulates squared gradients and divides the learning rate by their square root.

## Cell 30 — AdaGrad (code)

Implements AdaGrad and visualizes it. Three panels. Left: vanilla GD on the ravine. Middle: AdaGrad on the ravine — immediately you see it adapts per-parameter and takes a much more direct path. Right: effective learning rate for each parameter over time — you can see both decaying, but the steep-direction parameter's LR drops faster than the shallow one. The printed caveat: AdaGrad accumulates forever, so its LR can shrink too much and effectively stop learning.

## Cell 31 — RMSprop (markdown)

RMSprop fixes AdaGrad's shrinking-LR problem by using an exponential moving average of squared gradients instead of an accumulation. This way old gradients are gradually forgotten, so the LR can go both up and down. Works much better for deep learning where the landscape changes during training.

## Cell 32 — AdaGrad vs RMSprop (code)

Three panels. Left: AdaGrad's path on the ravine — decays too fast, fails to reach the minimum. Middle: RMSprop's path — maintains effective steps throughout and actually reaches the minimum. Right: loss curves showing AdaGrad plateauing while RMSprop keeps dropping.

## Cell 33 — Adam (markdown)

Adam combines momentum (first moment — running average of gradients) with RMSprop (second moment — running average of squared gradients), plus a bias correction term to fix the fact that both running averages start at zero. The default hyperparameters — learning rate 0.001, β1 = 0.9, β2 = 0.999, ε = 1e-8 — work remarkably well for a huge range of problems. Bias correction is especially important in early steps when the running averages are still small.

## Cell 34 — Grand optimizer comparison (code)

The money shot of the notebook. We run five optimizers — vanilla GD, SGD with momentum, AdaGrad, RMSprop, and Adam — all from the same starting point on the same ravine. Each gets its own panel showing its path, with the final loss labeled. The sixth panel stacks all five loss curves on a log scale. You can directly see that Adam converges fastest and to the lowest loss, with momentum and RMSprop close behind, and plain GD far behind. This cell is the case for why Adam is the default choice for most deep learning.

## Cell 35 — AdamW (markdown)

Explains AdamW. The problem with combining L2 regularization and Adam: Adam scales *all* gradients by its adaptive learning rate, which means it also scales your regularization, making its strength inconsistent. AdamW decouples the two — it applies weight decay *after* the Adam update, not inside the gradient. The result is more stable regularization. AdamW is now the standard for transformers — BERT, GPT, LLaMA all use it.

## Cell 36 — Learning-rate schedules intro (markdown)

Explains why fixed learning rates are suboptimal. At the start, you can afford a large LR to make fast progress. Near the end, a large LR prevents settling into a fine minimum. The solution: change the learning rate over time. Common schedules: step decay, exponential decay, cosine annealing, linear warmup, and one-cycle policy.

## Cell 37 — Visualizing LR schedules (code)

A 2-by-3 grid plotting six schedules over 100 epochs. Step decay drops in discrete steps. Exponential decay smoothly halves. Cosine annealing follows a half-cosine curve ending near zero. Linear warmup ramps up then stays flat. Warmup plus cosine decay combines the two — used in BERT and GPT. One-cycle policy: warmup for 30%, then aggressive cosine decay. Each shows the classic look of that schedule.

## Cell 38 — Practical PyTorch comparison intro (markdown)

Sets up the real-world comparison: train the same network on the two-moons classification task with five different PyTorch optimizers and see who wins.

## Cell 39 — Training with five optimizers (code)

We define a simple three-layer network, train it for 100 epochs with SGD (LR 0.1), SGD + Momentum, Adam at LR 0.01, AdamW at LR 0.01 with weight decay 0.01, and RMSprop at LR 0.01. Each trial prints its final loss and accuracy.

## Cell 40 — Visualizing the comparison (code)

Three panels. Left: training loss curves for all five on a log scale — Adam and AdamW converge fastest, SGD is slowest, RMSprop in the middle. Middle: training accuracy over epochs — same ranking. Right: the learned decision boundary for AdamW overlaid on the moons dataset — a clean curved separation between the two classes. The conclusion printed at the bottom: Adam and AdamW typically converge faster, but SGD with momentum and a good schedule can match them on final accuracy.

## Cell 41 — Which optimizer to use (markdown)

Decision guide as an ASCII flowchart and table. Fine-tuning a pretrained model? AdamW at LR 1e-5. Training a CNN from scratch? SGD + Momentum at LR 0.1 with step decay. Transformers or Vision Transformers? AdamW at LR 1e-4 with warmup. GANs? Adam at LR 2e-4. Quick experiments? Adam at LR 1e-3.

## Cell 42 — Modern and advanced optimizers (markdown)

Introduces three state-of-the-art optimizers. **SAM (Sharpness-Aware Minimization)** — looks for flat minima by perturbing the weights adversarially before each step. Flat minima generalize better. **Lion** — discovered by Google via neural architecture search on optimizers. Uses the sign of the momentum instead of its magnitude — simpler and more memory-efficient. **LARS / LAMB** — per-layer adaptive learning rates, used for extremely large-batch training.

## Cell 43 — Sharp vs flat minima (code)

Two plots. Left: a loss with a sharp minimum — a tiny change in W causes a large change in loss, so it generalizes poorly. Right: a smooth flat minimum — small changes in W barely change the loss, so it's robust to perturbations and generalizes well. This is the visual intuition behind SAM: push the optimizer toward flatter regions.

## Cell 44 — Optimizer evolution timeline and summary (markdown)

A timeline from Cauchy's gradient descent in 1847 through SGD, Momentum, AdaGrad, RMSprop, Adam, AdamW, SAM, and Lion. The summary restates the eight big lessons: GD as the foundation, SGD's stochasticity helping escape minima, momentum adding physics, adaptive methods, Adam combining both, AdamW fixing weight decay, LR schedules being crucial, and modern optimizers pushing the frontier. Final recommendations: Adam to start, AdamW for transformers, SGD+Momentum for CNNs, SAM for best generalization, Lion for memory-constrained scenarios.

---

## Outro

That's optimizers end to end. The short version: default to Adam or AdamW for almost anything; use SGD with momentum when you care about squeezing out the last half-percent of CNN accuracy; always schedule your learning rate; and watch out for the classic pitfalls like ill-conditioning and saddle points. Thanks for watching!
