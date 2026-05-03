# Video Script: Convolutional Neural Networks from First Principles

**Notebook:** `final_cnn_fundamentals_tutorial.ipynb`  
**Author:** Nitish Chowdary Ratakonda  
**Estimated read time:** ~22–26 minutes

> Read this script aloud while scrolling through the executed notebook. Each section maps one-to-one to a cell.

---

## Intro

Hey everyone, I'm Nitish. In this video I'm walking through my executed Colab on **Convolutional Neural Networks from first principles**. CNNs are the architecture that revolutionized computer vision — from LeNet in 1998 to AlexNet in 2012 to modern architectures today. We'll build one from scratch, understand exactly how convolution and pooling work, and then train a real CNN on CIFAR-10. Let's go cell by cell.

---

## Cell 0 — Colab badge (markdown)

The "Open in Colab" badge at the top. Clickable link to run the notebook live. Skip.

## Cell 1 — Title and Table of Contents (markdown)

Title: *"Convolutional Neural Networks from First Principles."* This slide has a milestones table — LeNet-5 in 1998, AlexNet in 2012, VGGNet in 2014, ResNet in 2015, EfficientNet in 2020 — and the seven chapters we'll cover: the convolution operation, building CNN layers, pooling, CNN architectures, training, visualizing what CNNs learn, and a complete CIFAR-10 classifier.

## Cell 2 — Setup and imports (code)

Standard imports — PyTorch, torchvision, NumPy, Matplotlib, PIL. We set the device to GPU if available, print the PyTorch and torchvision versions, and fix the random seeds. The output just confirms the versions and the device being used.

## Cell 3 — Why not fully connected for images? (markdown)

This is the motivating question. A 224-by-224 RGB image has over 150 thousand pixels. A single fully-connected layer with 1000 neurons would need 150 *million* parameters just for the first layer. CNNs solve this with three ideas: **local connectivity** (each neuron only sees a small patch), **parameter sharing** (the same filter slides across the whole image), and **translation equivariance** (detecting a pattern works no matter where it is).

## Cell 4 — Manual convolution implementation (code)

We write `conv2d_manual` from scratch — a Python function that slides a small kernel across a bigger image, multiplying element-wise and summing at each position. Then we build a tiny 5-by-5 test image where the left half is 10s and the right half is zeros — basically a hard vertical edge. We define a Sobel-like edge-detection kernel, apply our convolution, and print the output. The result is a 3-by-3 matrix with big values (30) in the middle columns, showing the kernel clearly detected the vertical edge.

## Cell 5 — Step-by-step convolution visualization (code)

This cell visualizes the sliding-window idea. It plots eight snapshots in a 2-by-4 grid, each showing the kernel positioned at a different location on the image, with a red rectangle highlighting the current patch and the computed output value labeled on top. This is the clearest way to see *how* convolution actually works — one small window at a time.

## Cell 6 — Classic image-processing kernels (code)

We define a dictionary of eight famous hand-designed kernels: identity, edge detection, sharpen, box blur, Gaussian blur, Sobel X for vertical edges, Sobel Y for horizontal edges, and emboss. We build a synthetic test image with a square, a rectangle, and a gradient, then apply each kernel and display the result in a 3-by-3 grid. You can clearly see the edge detectors highlight boundaries, the blur kernels smooth things out, and emboss adds that classic 3D look. The big takeaway printed at the end: these are hand-designed, but CNNs *learn* their own optimal kernels from data.

## Cell 7 — PyTorch's Conv2d layer (markdown)

Explains the `nn.Conv2d` API in detail: input channels, output channels (number of filters), kernel size, stride, padding, and bias. Gives the output-size formula — `(H_in + 2·padding − kernel_size) / stride + 1` — and a table of common patterns: stride 1 with padding 1 keeps the size, stride 2 halves it, no padding shrinks by 2.

## Cell 8 — Conv2d in practice (code)

We instantiate a Conv2d with one input channel, three output channels, kernel size 3, stride 1, padding 1. We print its configuration and its weight-tensor shape — which is `(3, 1, 3, 3)`, reading as "3 output filters, each with 1 input channel and a 3-by-3 kernel." We pass a 28-by-28 grayscale image through it and confirm the output is also 28-by-28 because padding 1 preserves the size.

## Cell 9 — Padding and stride effects (code)

We run six different Conv2d configurations on the same 32-by-32 input and print each output size. "No padding" shrinks to 30. "Same padding" stays at 32. Stride 2 drops to 16. A 5-by-5 kernel with padding 2 preserves size. Add stride 2 and it halves. A 7-by-7 with stride 2 halves as well. This teaches you how to control the output size of any conv layer.

## Cell 10 — Multi-channel convolution (code)

This cell explains what convolution looks like on RGB images. An ASCII diagram shows that each filter actually has three sub-kernels — one for R, G, and B — which are summed to produce a single output feature map. We instantiate a Conv2d with 3 input channels and 32 output channels, print the weight shape `(32, 3, 3, 3)`, and then visualize filter number zero by showing its R, G, and B sub-kernels as three separate small images, plus a combined magnitude view. This is the moment where you really see that a "filter" in a CNN is a small 3D tensor.

## Cell 11 — Pooling explained (markdown)

Explains why we pool: to reduce spatial size, to increase the effective receptive field, and to add a bit of translation invariance. Shows a small ASCII example of max pooling vs average pooling on a 4-by-4 input, producing 2-by-2 outputs.

## Cell 12 — Pooling layers in code (code)

We build a tiny 4-by-4 tensor and run three types of pooling: MaxPool2d, AvgPool2d, and AdaptiveAvgPool2d with output size 1 (which is *global* average pooling). The printed output shows MaxPool keeping the bright values, AvgPool smoothing them out, and global pooling collapsing the entire map to a single scalar — the mean of all sixteen numbers.

## Cell 13 — Visualizing pooling (code)

We generate a checkerboard image and apply six operations: original, MaxPool 2x2, AvgPool 2x2, MaxPool 4x4, AvgPool 4x4, and global average pooling. The output grid lets you visually compare how different pooling kernels preserve or smooth the checkerboard pattern, and how each shrinks the spatial dimensions.

## Cell 14 — Classic CNN pattern (markdown)

Explains the universal CNN pattern: repeated blocks of Conv → ReLU → Pool for feature extraction, followed by a Flatten and a few fully-connected layers for classification. It also introduces the feature pyramid idea: as depth increases, spatial size shrinks but the number of channels grows, and features become progressively more abstract — edges, then textures, then parts, then whole objects.

## Cell 15 — LeNet-5 implementation (code)

We implement **LeNet-5**, Yann LeCun's 1998 design for digit recognition, as a PyTorch Module. It starts with two Conv + Tanh + AvgPool blocks taking a 1×32×32 image down to 16×5×5, then flattens and runs through three fully-connected layers ending in 10 class logits. We print the architecture and total parameter count (around 60,000 — tiny by today's standards), pass in a random image, and confirm it outputs a 10-dimensional vector.

## Cell 16 — Modern CNN building blocks (code)

This cell introduces a modern **ConvBlock**: Conv → BatchNorm → ReLU, packaged neatly. BatchNorm, introduced in 2015, normalizes activations within each mini-batch and enables higher learning rates. We then define a **ModernCNN** using these blocks with three stages — 32, 64, 128 channels — each doing double convolution plus MaxPool plus Dropout2d. It ends with global average pooling followed by a single Linear layer to 10 classes. We print the architecture and confirm the forward pass works with a batch of 3-channel 32-by-32 images.

## Cell 17 — Tracing feature-map sizes (code)

We write a helper that walks through every layer of a model and prints how the tensor shape evolves. For LeNet-5 on a 1-channel 32-by-32 input, you see it go from 1×32×32 to 6×28×28, to 6×14×14 after the first pool, and so on. For the Modern CNN on a 3-channel 32-by-32 input, you see the "double channels, halve spatial" pattern clearly — 32×32 to 16×16 with 64 channels, to 8×8 with 128, ending at 1×1 after global pooling.

## Cell 18 — Data augmentation theory (markdown)

Explains data augmentation — the trick of creating fake new training samples by applying random transformations. Random crops give position invariance, horizontal flips give mirror invariance, color jitter handles lighting variation, rotation handles tilt, and random erasing handles occlusion.

## Cell 19 — Defining transforms (code)

We build two torchvision transform pipelines. The training pipeline does random crops with padding, random horizontal flips, random rotations up to fifteen degrees, color jitter, and then converts to tensor and normalizes with the CIFAR-10 mean and standard deviation. The validation pipeline does *no* augmentation — just tensor conversion and normalization, so results are deterministic. The output just lists all the transforms.

## Cell 20 — Visualizing augmentations (code)

We build a synthetic red-green square image and apply the augmentation pipeline eleven times, displaying the original plus eleven augmented versions in a 3-by-4 grid. Each one is shifted, flipped, rotated, or color-jittered a little differently. This is the clearest possible demonstration of why augmentation effectively multiplies your dataset size.

## Cell 21 — Loading CIFAR-10 (code)

We download CIFAR-10 through torchvision. It's 50,000 training images and 10,000 test images, 32-by-32 RGB, across ten classes — plane, car, bird, cat, deer, dog, frog, horse, ship, truck. We wrap them in DataLoaders with batch size 128. Then we display ten sample images with their class labels in a 2-by-5 grid, so you can actually see what CIFAR-10 looks like.

## Cell 22 — Training and evaluation functions (code)

We define three helpers. `train_epoch` does one pass through the training data — forward, compute loss, backward, optimizer step, and track running loss and accuracy. `evaluate` does the same but with `model.eval()` and `torch.no_grad()` — no gradients, no updates. `train_cnn` ties them together: it sets up cross-entropy loss and Adam, a ReduceLROnPlateau scheduler on validation accuracy, runs for a set number of epochs, tracks history, and reports best validation accuracy.

## Cell 23 — Training the modern CNN (code)

We instantiate the Modern CNN and call `train_cnn` for ten epochs at learning rate 0.001. The output is an epoch-by-epoch log showing training loss, training accuracy, validation loss, and validation accuracy. On CIFAR-10 with this small architecture, you'll typically see validation accuracy climb into the seventies or low eighties over ten epochs.

## Cell 24 — Plotting training history (code)

Two plots side by side. Left: training loss versus validation loss over epochs — both trending down, with the validation curve slightly above training, which is normal. Right: training and validation accuracy over epochs — both rising. This is how you visually check whether your model is learning and whether it's starting to overfit.

## Cell 25 — Visualizing what CNNs learn (markdown)

Sets up the theme: CNNs learn hierarchical features. Early layers detect edges and colors, middle layers combine those into textures, later layers recognize parts like eyes or ears, and the deepest layers represent whole objects.

## Cell 26 — Visualizing learned filters (code)

We extract the weights of the first convolutional layer and plot thirty-two of them in a 4-by-8 grid. Each filter is a 3×3×3 tensor, so we can show them in RGB. You'll notice many of them look like oriented edge detectors or color-opponent filters — which is remarkable because the network was never told to learn those; it discovered them from data. They look strikingly similar to classical Gabor filters used in image processing for decades.

## Cell 27 — Visualizing feature maps (code)

We register forward hooks on every Conv2d layer to capture its activations, then run a test image through the network. We then plot the feature maps layer by layer. In the early layers you see recognizable edge-and-texture-like responses. In later layers the maps become smaller and more abstract — you can't tell what they represent by eye, but they encode increasingly high-level features.

## Cell 28 — Production-ready CNN (markdown)

A short intro saying we're going to build a more sophisticated production-grade CNN next.

## Cell 29 — ProductionCNN implementation (code)

This is a much bigger architecture. Four stages with 64, 128, 256, and 512 channels, each doing two Conv + BatchNorm + ReLU blocks followed by MaxPool. Dropout2d between stages for regularization. Kaiming-normal initialization for the conv weights — the standard init for ReLU networks. Global average pooling plus a single Linear layer at the end. We print the architecture and total parameter count, and confirm forward-pass shapes.

## Cell 30 — Advanced training function (code)

This defines `train_advanced`, which layers several production-grade techniques on top of basic training: **AdamW** with weight decay instead of plain Adam, a **cosine-annealing** learning-rate schedule, **gradient clipping** to a max norm of 1 to stabilize training, **early stopping** with patience of 5 epochs, and **model checkpointing** — saving the best weights to disk. This is essentially what a real-world training loop looks like.

## Cell 31 — Training the production CNN (code)

We instantiate the ProductionCNN and call `train_advanced` for 15 epochs. The log shows each epoch's learning rate (which you can see decaying), training accuracy, validation accuracy, and running best. On CIFAR-10, this architecture typically reaches mid-to-high eighties validation accuracy even in 15 epochs.

## Cell 32 — Comprehensive results visualization (code)

A 2-by-2 figure. Top-left: loss curves. Top-right: accuracy curves. Bottom-left: the cosine-annealing learning-rate schedule — a smooth sinusoidal decay. Bottom-right: the "generalization gap," which is training accuracy minus validation accuracy over time. If that gap grows, you're overfitting; if it stays small, you're generalizing well.

## Cell 33 — Confusion matrix and per-class analysis (code)

We load the best saved model weights, run inference on the entire test set, and build a confusion matrix with sklearn. We plot it as a heatmap, so you can literally see which classes get confused with which — cats and dogs, or ships and planes, are classic confusions. Next to it, a bar chart shows per-class accuracy, colored green, orange, or red based on thresholds. Finally, sklearn's classification report prints precision, recall, and F1 for every class.

## Cell 34 — CNN summary (markdown)

Recaps the components we built: convolution, pooling, BatchNorm, Dropout, and global average pooling. Lists the five core CNN design rules: double channels when halving spatial size, prefer 3-by-3 kernels, BatchNorm after every conv, end with global average pooling, and augment the training data aggressively.

## Cell 35 — CNN cheat sheet (code)

The final cell prints a comprehensive cheat sheet with the PyTorch API for convolution, pooling, normalization, regularization, modern conv blocks, data augmentation, and the standard training loop. Useful as a reference to pin up while building your own CNNs.

---

## Outro

That's the complete CNN fundamentals walkthrough — from understanding what a single convolution does, all the way up to training a production-grade CNN on CIFAR-10 with modern techniques like BatchNorm, Dropout, cosine annealing, gradient clipping, and early stopping. In the next notebook we'll go even deeper into *modern* CNN architectures — ResNet, EfficientNet, and transfer learning. Thanks for watching!
