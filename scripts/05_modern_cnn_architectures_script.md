# Video Script: Modern CNN Architectures — ResNet to EfficientNet

**Notebook:** `final_modern_cnn_architectures_tutorial.ipynb`  
**Author:** Nitish Chowdary Ratakonda  
**Estimated read time:** ~20–24 minutes

> Read this script aloud while scrolling through the executed notebook. Each section maps one-to-one to a cell.

---

## Intro

Hey everyone, I'm Nitish. In this video I'm walking through my executed Colab on **modern CNN architectures** — the innovations that took us from training 20-layer networks being barely possible to training 150-layer networks being easy, and from handcrafted architectures to compound-scaled ones like EfficientNet. We'll also cover transfer learning. Let's go cell by cell.

---

## Cell 0 — Colab badge (markdown)

The Colab badge. Click to launch.

## Cell 1 — Title and timeline (markdown)

Title: *"Modern CNN Architectures — From ResNet to EfficientNet."* Shows the architecture timeline: AlexNet 2012 with 8 layers and 16.4% error, VGG 2014 with 19 layers at 7.3%, GoogLeNet with Inception modules, ResNet 2015 with 152 layers and 3.6% — the single biggest jump in the chart — then DenseNet, EfficientNet, and ConvNeXt. The table of contents covers seven chapters: the vanishing-gradient problem, ResNet, advanced blocks, EfficientNet, transfer learning, fine-tuning strategies, and a complete real-world application.

## Cell 2 — Setup and imports (code)

PyTorch, torchvision, and torchvision's `models` module for loading pretrained networks. Prints the PyTorch version, device, and — if a GPU is present — its name and memory. Sets random seeds.

## Cell 3 — The vanishing-gradient problem (markdown)

Explains why deep networks failed before 2015. During backprop, gradients multiply through layers. If each layer's gradient term is less than one — like sigmoid's max of 0.25 — then after 50 layers you have 0.25 to the 50th power, which is effectively zero. The earliest layers simply stop learning.

## Cell 4 — Demonstrating vanishing gradients (code)

We build 20-layer deep MLPs with three different activations: sigmoid, ReLU, and tanh. We do a single forward-backward pass and measure the gradient norm at every layer. The left plot on a linear scale shows the effect, but the right plot on a log scale makes it dramatic: with sigmoid, gradients vanish exponentially and are basically unusable at the bottom; tanh is better but still vanishes; ReLU keeps gradients roughly stable throughout — which is why ReLU was so transformative. The printed summary drives the point home.

## Cell 5 — ResNet and residual learning (markdown)

The key insight of ResNet from 2015. Instead of learning a mapping `H(x)` directly, the network learns the **residual** `F(x) = H(x) − x`, and the block outputs `F(x) + x`. The ASCII diagram shows the skip connection clearly. This gives three benefits: if the optimal mapping is the identity, the network just learns `F(x) = 0`; gradients flow directly through the skip connection like a highway; and later layers can reuse features from earlier ones.

## Cell 6 — Basic residual block implementation (code)

We implement `BasicResidualBlock` — the block used in ResNet-18 and ResNet-34. Two 3-by-3 convs, each with BatchNorm, with a ReLU in between, plus a skip connection from input to output. If the stride is 2 or the channel count changes, we project the shortcut through a 1-by-1 conv so dimensions match. The forward method does `out = out + self.shortcut(x)` — that one line is the key innovation. We test it with and without downsampling.

## Cell 7 — Bottleneck block (code)

We implement the **bottleneck block** used in ResNet-50, 101, and 152. The structure is 1×1 conv to squeeze channels down, 3×3 conv for spatial filtering, 1×1 conv to expand back up by 4x, plus the skip connection. The print statement compares parameters: a basic block at 256 channels needs vastly more parameters than a bottleneck at the same output channels. That's why ResNets of 50 layers and up use bottlenecks — they're dramatically more parameter-efficient.

## Cell 8 — Complete ResNet (code)

We implement `ResNet` as a general class that takes a block type and a list of layer counts. Standard stem — 7×7 conv stride 2, BatchNorm, MaxPool — then four residual stages with 64, 128, 256, and 512 channels, and a final average-pool plus linear classifier. We define convenience constructors for ResNet-18, 34, 50, 101, and 152. The printed table shows each variant's parameter count — from about 11 million for ResNet-18 up to around 60 million for ResNet-152.

## Cell 9 — Gradient flow: plain vs residual (code)

This is the experimental proof that skip connections fix vanishing gradients. We build two 20-block deep networks — one with plain blocks and one with residual blocks — push a single batch through, do backprop, and plot the gradient norm at every conv layer for both. On the linear plot the difference is visible; on the log plot it's dramatic. The plain network's early gradients are orders of magnitude smaller than the residual network's. The skip connections really do create a "gradient highway."

## Cell 10 — Advanced blocks (markdown)

Intro to three modern building blocks: **SE blocks** for channel attention, used in SENet and EfficientNet. **MBConv**, the mobile inverted bottleneck used in MobileNet and EfficientNet. And **ConvNeXt blocks**, a modernized ResNet design.

## Cell 11 — SE block implementation (code)

The Squeeze-and-Excitation block. Given an input with C channels, "squeeze" via global average pooling to a C-dimensional vector, pass it through a small bottleneck MLP — FC down to C/16, ReLU, FC back up to C — and a sigmoid to get per-channel weights, then multiply the original input by those weights. Essentially, the block learns which channels are currently important and rescales them. We build one for 64 channels and print its parameter count — it adds only a tiny fraction of parameters to a normal conv.

## Cell 12 — MBConv block (code)

We implement MBConv — the mobile inverted bottleneck from MobileNetV2 and EfficientNet. The structure is **opposite** of a regular bottleneck: we *expand* channels via a 1×1 conv, do a depthwise 3×3 conv for spatial filtering (which is much cheaper because each output channel comes from exactly one input channel), optionally apply an SE block, then project back down with another 1×1 conv. The activation is Swish / SiLU, and there's no ReLU after the projection — a "linear bottleneck." We compare parameter counts against a standard two-conv block — MBConv is much more efficient for the same input/output shape.

## Cell 13 — EfficientNet theory (markdown)

Explains compound scaling, EfficientNet's main contribution. Instead of scaling depth, width, and resolution independently, EfficientNet scales them together using a compound coefficient φ, with constraints that keep FLOPS growing in a predictable way. This gives a principled way to make a family of models — B0 through B7 — each balancing those three dimensions.

## Cell 14 — EfficientNet configuration table (code)

Prints the EfficientNet-B0 baseline architecture — seven stages with increasing channels and decreasing spatial size — and the scaling coefficients for variants B0 through B7. B0 uses 224×224 input, B4 uses 380×380, B7 uses 600×600, with width and depth multipliers scaled accordingly.

## Cell 15 — Loading pretrained EfficientNet (code)

We load pretrained EfficientNet-B0 and B4 from torchvision with their ImageNet weights. Print the parameter counts and ImageNet top-1 accuracies: B0 has about 5 million parameters and 77% accuracy, B4 has about 19 million and 83%. For comparison, ResNet-50 has 25 million parameters and only 76% accuracy — so EfficientNet is strictly better on accuracy-per-parameter.

## Cell 16 — Transfer learning intro (markdown)

Explains transfer learning — instead of training from scratch (which needs millions of images, expensive GPUs, and careful tuning), we start from a model pretrained on ImageNet and adapt it. Two approaches: **feature extraction**, where you freeze the backbone and only train the classifier head — good for small datasets in similar domains. And **fine-tuning**, where you let the whole network continue to learn at a low learning rate — better for larger datasets or different domains.

## Cell 17 — Transfer model builder (code)

We define `create_transfer_model`. It loads a pretrained ResNet-18, ResNet-50, or EfficientNet-B0, replaces the final classification head with a new Linear layer matching our number of classes, and optionally freezes the backbone. We create two versions — one with the backbone frozen and one fully trainable — and print the total and trainable parameter counts. With freezing, only a tiny fraction — often under 1% — of parameters are trainable.

## Cell 18 — Transforms for transfer learning (code)

We set up the standard ImageNet transforms. Training: resize to 256, random crop to 224, random horizontal flip, color jitter, then tensor conversion and normalization with the ImageNet mean and standard deviation. Validation: resize to 256, center crop to 224, tensor, normalize. The ImageNet mean and standard deviation values are critical — pretrained models *expect* these exact normalization statistics.

## Cell 19 — Loading CIFAR-10 for transfer (code)

For the demo we use CIFAR-10 resized to 224. We take a subset of 5,000 training and 1,000 test images to keep the demo fast, and wrap in DataLoaders. In a real project you'd use your own custom dataset, but the code structure is identical.

## Cell 20 — Fine-tuning strategies (markdown)

Lists three advanced techniques: **gradual unfreezing** (start with just the head, then unfreeze deeper layers one stage at a time), **discriminative learning rates** (use a lower LR for earlier layers than for later layers), and **warm restarts** (cyclical learning rate schedules).

## Cell 21 — Discriminative learning rates (code)

Implements `get_param_groups_with_discriminative_lr`. It creates separate optimizer parameter groups for the classifier (highest learning rate), then layer4, layer3, layer2, layer1, and the stem — each with ten times less learning rate than the next. The printed table shows exactly what LR each layer gets. This preserves the pretrained low-level features while letting the task-specific layers adapt quickly.

## Cell 22 — Transfer-learning training function (code)

Defines `train_transfer_model`. It moves the model to the device, sets up Adam on only the trainable parameters, schedules with cosine annealing, and runs the standard train/validate loop for a set number of epochs, tracking history at each epoch.

## Cell 23 — Comparing three approaches (code)

The big experiment. We train three models for 5 epochs each on the same CIFAR-10 subset: **feature extraction** (frozen ResNet-18 backbone, only classifier trained), **fine-tuning** (same architecture, all layers trainable, lower LR of 1e-4), and **from scratch** (no pretrained weights). The printed logs let you watch each approach train.

## Cell 24 — Visualizing the comparison (code)

Two plots. Left: validation accuracy versus epoch for all three approaches. You'll see feature extraction and fine-tuning both start high and climb — fine-tuning typically ends up a bit higher — while training from scratch starts near random and claws its way up slowly. Right: a bar chart of final accuracies making the gap obvious. The insights printed at the end: pretrained features are extremely valuable, transfer learning converges much faster, and fine-tuning usually wins when you have enough data.

## Cell 25 — Modern CNN summary (markdown)

Wrap-up table: skip connections enabled deep networks in ResNet. Bottleneck blocks reduced parameters. SE blocks added channel attention. MBConv gave mobile-efficient blocks. Compound scaling balanced depth, width, and resolution in EfficientNet. Transfer-learning guidelines: for a small dataset, use feature extraction; for medium, fine-tune the top; for large with a different domain, fine-tune everything with discriminative LRs.

## Cell 26 — Modern CNN cheat sheet (code)

Prints a cheat sheet covering residual blocks (basic and bottleneck), the SE block, MBConv, how to load pretrained torchvision models, the standard ImageNet normalization statistics, transfer-learning strategies, and quick model-selection guidance based on your deployment target.

---

## Outro

That's modern CNN architectures from vanishing gradients to transfer learning. The key takeaways: skip connections made deep networks possible, bottleneck and depthwise-separable blocks made them efficient, EfficientNet gave us principled scaling, and transfer learning lets us leverage all of that pretrained knowledge for our own tasks with very little data. Thanks for watching!
