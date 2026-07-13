# Deep Learning — Exam Revision Notes

---

## 1. What is Deep Learning?

- Deep Learning (DL) = subfield of ML that uses **artificial neural networks with many layers** to automatically learn patterns directly from raw data — no hand-crafted features needed.
- Thrives on **big data** — performance keeps improving as data grows.
- Powered by **GPUs** — parallel hardware made deep nets practical.
- Learns features itself — hierarchical layers replace manual feature engineering.

---

## 2. AI vs ML vs DL

- Relationship: **Artificial Intelligence → Machine Learning → Deep Learning** (each is a subset of the one before it).
- **AI** — systems performing tasks that typically need human intelligence (reasoning, planning, decision-making).
- **ML** — subset of AI; learns patterns from data without being explicitly programmed.
- **DL** — subset of ML; uses multi-layered neural networks to automatically learn complex representations from data.

---

## 3. DL vs ML — Comparison

| Aspect | Machine Learning | Deep Learning |
|---|---|---|
| Data dependency | Works well with smaller data | Needs large data; improves as data grows |
| Hardware dependency | Can run on a normal CPU | Needs GPUs for large matrix computations |
| Training time | Faster training; prediction can be slow (e.g., KNN) | High training time; faster prediction |
| Feature selection | Manually selected by a domain expert | Extracted automatically by the network |
| Interpretability | More interpretable (e.g., Decision Tree, Random Forest) | Acts more like a "black box" |

---

## 4. Why Deep Learning Became Popular

1. **Big Data** — the digital age produced huge datasets; DL models thrive on and need vast data to train effectively.
2. **Hardware (GPU)** — GPUs offer hundreds of cores handling thousands of parallel threads (vs. CPUs' few cores), making training on millions/billions of parameters feasible; affordable high-performance GPUs boosted adoption.
3. **Software** — breakthroughs in architectures (Transformers for language, CNNs for vision) and frameworks (PyTorch, TensorFlow) made deep networks easier to build and train effectively.

---

## 5. Biological Neuron vs Artificial Neuron

| Biological Neuron | Artificial Neuron |
|---|---|
| Dendrites — receive signals from other neurons | Inputs (x) — feature values fed into the unit |
| Cell body (soma) — integrates incoming signals | Weights (w) — learned importance of each input |
| Axon — carries the output signal onward | Summation — weighted sum plus a bias term |
| Synapse — connection strength between neurons | Activation — non-linear function gives the output |

- Human learning: observing experiences → feedback → forming memories → strengthening/weakening neural connections.
- DL model "learning": input data → prediction → calculate error → adjust weights (via gradient descent) → repeat over iterations — inspired by **synaptic plasticity**.

---

## 6. The Perceptron

### 6.1 What is a Perceptron?

- The **perceptron** is the simplest unit of a neural network — the structural building block of a deep learning model.
- It's a mathematical model of a single neuron: takes multiple inputs, processes them, and produces a single output.
- Basic idea: receives input values → multiplies each by a weight → adds a bias → passes the result through an activation function → produces an output (often 0 or 1).

### 6.2 How a Perceptron Works

- Three steps:
  1. Takes inputs (x₁, x₂, ... xₙ) — features/signals.
  2. Multiplies each by a weight (w₁, w₂, ... wₙ) — weights represent importance.
  3. Adds a bias, then applies an activation function to decide the output.
- Core formula (conceptual):

  **ŷ = f(Σ wᵢxᵢ + b)**

  - `xᵢ` = inputs, `wᵢ` = weights, `b` = bias, `f` = activation function, `ŷ` = output/prediction.

### 6.3 Types of Perceptron

- **Single-Layer Perceptron** — one input layer connected directly to output; can only solve **linearly separable** problems (e.g., AND, OR, NAND gates).
- **Multi-Layer Perceptron (MLP)** — has hidden layers between input and output; can solve complex, **non-linear** problems (e.g., XOR, image recognition).

### 6.4 Gates in a Perceptron (Decision-Making)

- A **gate** is a decision-making unit that takes one or more binary inputs (0 or 1) and produces exactly one binary output, based on a fixed logical rule.
- In a perceptron, a gate is implemented by choosing the right weights and bias so the neuron's step function mimics that logical rule.
- General structure: **z = Σ(wᵢ × xᵢ) + b**, then **output = 1 if z ≥ 0, else 0**.

### 6.5 Types of Logic Gates

- **AND** — "both required"; fires only when every input is active.
- **OR** — "at least one"; fires if any input is on.
- **NOT** — simplest gate; one input, flips it (an inverter).
- **NAND** — "Not AND"; an AND gate with its output inverted; considered a **universal gate**.
- **XOR (Exclusive OR)** — "one but not both"; outputs 1 only when inputs differ. Historically important — a single perceptron **cannot** learn XOR.

### 6.6 Why Gates Matter in Neural Networks

- Gates are the **foundation of computation** — every computer chip is built from combinations of basic logic gates.
- Each perceptron acts like a gate — it fires (1) or stays silent (0) based on its inputs; **stacking perceptrons = deep learning**.
- The **XOR problem** proved a single perceptron is limited — this limitation drove the invention of **multi-layer networks (MLP)** and, eventually, deep learning.

### 6.7 Decision Boundary

- A **decision boundary** is the line, curve, or surface that separates different classes in a classification problem — the "border" where a model's prediction switches from one class to another.
- For a perceptron, it's a straight line (or hyperplane) separating output-0 region from output-1 region.

### 6.8 The XOR Problem

- No single straight line can separate XOR's classes correctly — the two "same output" pairs (0,0)/(1,1) and the two "different output" pairs (0,1)/(1,0) can't be split by one line.
- This is called **not linearly separable**.
- Since a single perceptron can only draw one straight-line decision boundary, **XOR is impossible for a single perceptron to solve** — this is exactly why Multi-Layer Perceptrons (with hidden layers) are needed.

---

## 7. Structure of a Deep Neural Network

- A DNN typically consists of:
  - **Input Layer** — receives raw data.
  - **Hidden Layers** — extract increasingly complex features.
  - **Output Layer** — produces predictions/classifications.
- "**Deep**" simply means many hidden layers stacked between input and output (more than one hidden layer).
- Example (image recognition, cat detection): 1st hidden layer detects edges/corners → 2nd detects shapes/textures → 3rd recognizes eyes/ears/whiskers → output layer classifies as "cat." This hierarchical learning resembles the human visual cortex.

---

## 8. Activation Functions

### 8.1 What is an Activation Function?

- Each neuron forms a weighted sum of its inputs and passes the result to a function called the **activation function** (also called transfer function).
- Formula (conceptual): **a = g(w₁x₁ + w₂x₂ + ... + wₙxₙ + b)** — `g()` determines how strongly a neuron "fires."

### 8.2 Why is an Activation Function Required?

- Introduces **non-linearity**, letting networks model complex relationships that plain stacked linear operations never could.
- Without it: stacking multiple linear layers still collapses to a **single linear equation**, no matter how many layers — the network gains no extra power from depth.
- With it: each layer can learn a genuinely new representation, enabling networks to model real-world (non-linear) data — this is what makes **universal approximation** possible.

### 8.3 Ideal Activation Function — Properties

1. **Non-linear** — needed to handle complex, real-world patterns; without it, deep networks reduce to a single linear model.
2. **Differentiable** — must have a computable derivative, since gradients are used during backpropagation to update weights.
3. **Computationally inexpensive** — evaluated millions of times per training pass; expensive functions (like exponentials) slow training significantly.
4. **Zero-centered (normalized)** — output ideally centered around mean ≈ 0, leading to faster convergence (e.g., Tanh).
5. **Non-saturating** — shouldn't "squeeze"/saturate the input range; saturating functions (Sigmoid, Tanh) cause the **Vanishing Gradient Problem**.

### 8.4 Types of Activation Functions

| Function | Range | Notes |
|---|---|---|
| Sigmoid | [0, 1] | Squashes values; good for probabilities (binary classification output) |
| Tanh | [−1, 1] | Zero-centered outputs; used in hidden layers |
| ReLU | [0, ∞) | max(0, x) — fast, default choice for most hidden layers |
| Leaky ReLU | (−∞, ∞) | Fixes "dying ReLU" neurons |
| ELU | (−α, ∞) | Smooth negative values, more robust |

### 8.5 Sigmoid (Logistic) Function

- Formula: **σ(z) = 1 / (1 + e⁻ᶻ)**, Range: [0, 1].
- Advantages: smooth and differentiable everywhere; output interpretable as a probability; good for binary classification output layer.
- Disadvantages: saturating function → **vanishing gradient**; not zero-centered → slower convergence; computationally expensive (exponential).

### 8.6 Tanh (Hyperbolic Tangent) Function

- Formula: **f(x) = (eˣ − e⁻ˣ) / (eˣ + e⁻ˣ)**, Range: [−1, 1].
- Advantages: zero-centered (unlike Sigmoid) → more balanced gradients, faster convergence than Sigmoid.
- Disadvantages: still a saturating function → still prone to vanishing gradient at extremes; computationally expensive.

### 8.7 ReLU (Rectified Linear Unit)

- Formula: **f(x) = max(0, x)**, Range: [0, ∞).
- Key idea: keep it if positive, make it zero if negative.
- Advantages: handles non-linear data well; not saturated on the positive side; computationally inexpensive; faster convergence.
- Disadvantages: not differentiable at x = 0; not zero-centered; the **Dying ReLU Problem**; loses information for negative values.

### 8.8 The Dying ReLU Problem

- If a neuron keeps receiving negative inputs during training, ReLU converts them all to 0 — the neuron's gradient becomes 0, so its weights stop updating.
- The neuron becomes permanently inactive ("dead") and stops learning entirely.
- **Solution: Leaky ReLU** — allows a small non-zero slope for negative inputs, so the neuron always has a non-zero gradient and can keep learning.

### 8.9 Leaky ReLU & ELU

- **Leaky ReLU**: f(x) = x if x > 0, else 0.01x. Removes the dying ReLU problem; computationally inexpensive; but the negative slope (0.01) is fixed, not learned.
- **ELU (Exponential Linear Unit)**: f(x) = x if x > 0, else α(eˣ − 1). Smooth negative activations, zero-centered outputs, reduces vanishing gradient; but computationally more expensive (exponential) and saturates for large negative values.

### 8.10 Vanishing Gradient Problem — Deep Dive

- During backpropagation, gradients are multiplied across layers; if each layer's gradient is small (< 1), repeated multiplication shrinks it exponentially toward 0.
- Early layers then receive near-zero gradient updates and effectively stop learning.

| Activation | Vanishing Gradient? | Reason |
|---|---|---|
| Sigmoid | Yes | Max gradient only 0.25; squishes gradients heavily |
| Tanh | Yes | Better than Sigmoid, but still saturates at extremes |
| ReLU | Partial | Positive side fine; negative side = 0 gradient (dying) |
| Leaky ReLU | No | Always has a non-zero gradient |
| ELU | No | Smooth negative activations; no dead neurons |

### 8.11 Complete Comparison Table

| Function | Formula | Range | Zero-Centered | Saturating | Vanishing Grad | Comp. Cost | Best Use |
|---|---|---|---|---|---|---|---|
| Sigmoid | 1/(1+e⁻ᶻ) | [0, 1] | ✗ | Yes | Yes | High | Binary output |
| Tanh | (eˣ−e⁻ˣ)/(eˣ+e⁻ˣ) | [−1, 1] | ✓ | Yes | Yes | High | Hidden layers |
| ReLU | max(0, x) | [0, ∞) | ✗ | No (pos side) | Partial | Low | Most hidden layers |
| Leaky ReLU | x or 0.01x | (−∞, ∞) | ✗ | No | No | Low | Fix dying ReLU |
| ELU | x or α(eˣ−1) | (−α, ∞) | ~Yes | Partial | No | Medium | Robust training |

- **General rule of thumb**: Use ReLU as default → switch to Leaky ReLU/ELU if dying ReLU occurs → Sigmoid only for binary output layer → Tanh for zero-centered hidden layers.

---

## 9. Training Process (How a Network Learns)

- Learning is an iterative loop — predict, measure error, then adjust weights to improve the next prediction:
  1. **Forward Pass** — input flows through the layers to produce a prediction.
  2. **Loss Function** — compares prediction to the true value and scores the error.
  3. **Backpropagation** — propagates the error backward to determine each weight's "blame."
  4. **Gradient Descent** — updates weights in the direction that reduces the loss.
- This loop repeats over many **epochs** until the loss stops improving — that's "training."

---

## 10. Key Terms

- **Weights & Bias** — trainable parameters the network adjusts as it learns.
- **Epoch** — one full pass through the entire training dataset.
- **Learning Rate** — step size controlling how fast weights update.
- **Loss / Cost** — a number measuring how wrong the predictions are.
- **Overfitting** — memorizing training data; fails on new/unseen data.
- **Generalization** — performing well on unseen, real-world data.

---

## 11. Tensors

### 11.1 What is a Tensor?

- A **tensor** is a multi-dimensional grid of numbers — the fundamental data structure for ML/DL models.
- Generalizes across dimensions:
  - 0D tensor = single number (scalar)
  - 1D tensor = list (vector)
  - 2D tensor = table (matrix)
  - 3D+ tensor = nested grids of data

### 11.2 Tensor vs Array

- **Hardware acceleration** — standard arrays process math on the CPU (line-by-line); tensors are built to run on GPUs/TPUs, enabling massively parallel computation (much faster for large-scale ML math).
- **Automatic differentiation** — plain arrays just store numbers with no memory of how they were computed; tensors keep a running history of operations performed on them, allowing automatic computation of gradients needed for backpropagation/training.

---

## 12. TensorFlow vs PyTorch

- **TensorFlow** — developed by Google Brain; strong in production deployment (TensorFlow Serving, Lite, TFX) and offers Keras as a simple high-level API.
- **PyTorch** — developed by Meta AI; Python-first, dynamic computation graph, dominant in research and Generative AI, tightly integrated with Hugging Face.

| Aspect | TensorFlow | PyTorch |
|---|---|---|
| Core philosophy | Production-focused, structured, robust | Python-first, flexible, imperative |
| Graph execution | Static (by default) — compiles graph before running | Dynamic — builds computational graph on the fly |
| Debugging | Harder — needs specialized tools (TensorBoard) | Easy — works with standard Python debuggers |
| Ecosystem strength | Enterprise production, industrial automation | Academic research, Generative AI, Hugging Face |
| Mobile/Edge support | Industry-best (TensorFlow Lite) | Good (PyTorch Edge), historically younger |

- **Core similarity**: both use tensors as their core data structure, support hardware acceleration (GPU/TPU), automatic differentiation, high-level APIs (Keras / `torch.nn`), and are production-deployment ready.

---

## 13. Transfer Learning

- Uses **existing, pre-trained architectures** instead of building models from scratch — reduces time, effort, and money.
- Common pre-trained models by task:

| Task | Pre-trained Model |
|---|---|
| Image classification | ResNet |
| Text | BERT |
| Image segmentation | U-Net |
| Image translation | Pix2Pix |
| Object detection | YOLO |
| Speech recognition | WaveNet |

---

## 14. Real-World Applications of Deep Learning

- **Computer Vision** — image classification, object detection, medical imaging.
- **Natural Language Processing** — translation, chatbots, large language models.
- **Speech & Audio** — voice assistants, transcription, audio generation.
- **Autonomous Systems** — self-driving perception, robotic control.
- **Healthcare** — diagnosis support, drug discovery pipelines.
- **Generative AI** — creating images, text, code, and music.

---

## 15. Hardware for Deep Learning

- Big matrix operations on a CPU are slow, so DL relies on parallel-processing hardware:
  - **GPU** — great for big matrix math and image work (NVIDIA/CUDA).
  - **FPGA** — programmable, fast, low-power, re-programmable (but expensive).
  - **ASIC/TPU** — application-specific chip; Google's Tensor Processing Unit.
  - **NPU** — dedicated silicon for running neural networks efficiently on devices.

---

## 16. Optimization & Gradient Descent

### 16.1 What is Optimization in Deep Learning?

- Training a neural network means **minimizing a loss function** — a value measuring how wrong the model's predictions are.
- The most common approach to this minimization is **Gradient Descent (GD)**, which updates parameters by following the negative of the gradient.
- Analogy: a traveler descending a valley in the dark — they feel the slope underfoot (the gradient) and step in the direction that goes downhill (reduces error).

### 16.2 Cost Function

- The **cost function** measures the overall error between the model's predictions and actual values across the dataset.
- For regression tasks, **Mean Squared Error (MSE)** is a commonly used cost function — it averages the squared difference between predicted and actual values (already covered as a regression metric).
- Goal of training: find the parameters that **minimize** the cost function.

### 16.3 What is Gradient Descent?

- Gradient Descent is the **core optimization algorithm** used to train neural networks (and many ML models).
- It works by iteratively adjusting model parameters in the direction that reduces prediction error.

**How it works (conceptual loop):**
1. **Forward Pass** — feed input through the network to get a prediction.
2. **Compute Loss** — measure how wrong the prediction is.
3. **Backward Pass (Backpropagation)** — compute gradients of the loss with respect to every weight.
4. **Update Weights** — nudge each weight slightly in the direction that reduces loss.
5. Repeat until the loss converges.

### 16.4 The Weight Update Rule

- Formula (conceptual): **w = w − η · ∇L(w)**
  - `w` = model weight, `η` (eta) = learning rate (step size), `∇L(w)` = gradient of the loss w.r.t. the weight.
- In words: **New Weight = Old Weight − Learning Rate × Gradient**.

### 16.5 Backpropagation

- **Backpropagation** is the algorithm used to train neural networks by computing the gradient of the loss with respect to each weight, applying the **chain rule** of calculus from the output layer back to the input layer.
- These gradients are then used by Gradient Descent to actually update the weights.
- Analogy: tracing blame backward through a company hierarchy after a loss — Director → Manager → Team Lead → Employee — each level gets "blamed" in proportion to how much it actually contributed to the final error; backpropagation does the same for every neuron/weight in a network.

### 16.6 The Vanishing Gradient Problem

- During backpropagation, gradients are **multiplied across layers** via the chain rule. If each layer's gradient is small (< 1), repeated multiplication shrinks the overall gradient exponentially, approaching 0.
- Early layers then receive near-zero gradient updates and effectively **stop learning**.
- This is especially common with **saturating activation functions** (Sigmoid, Tanh), since their derivatives are always less than 1.

### 16.7 Why ReLU Fixes the Vanishing Gradient Problem

- ReLU's derivative is exactly **1** for any positive input (not a fraction like Sigmoid/Tanh).
- So in a chain of ReLU multiplications, the gradient doesn't shrink — it passes through largely unchanged, allowing even early layers in a deep network to receive a strong, usable gradient signal.

### 16.8 Types of Gradient Descent

1. **Batch Gradient Descent** — computes the gradient using the **entire training dataset** before each update.
   - Advantages: stable, smooth convergence; easy to parallelize.
   - Disadvantages: high memory requirement; slow for large datasets; can get stuck in poor local minima.
2. **Stochastic Gradient Descent (SGD)** — updates parameters using **one training example at a time**.
   - Advantages: faster updates, frequent feedback on progress, noisy updates can help escape local minima, suitable for large datasets.
   - Disadvantages: noisy/fluctuating gradient signal; harder to settle precisely at the minimum.
3. **Mini-Batch Gradient Descent** — combines both ideas; splits data into small batches (commonly 50–256 samples) and updates after each batch.
   - Advantages: balances stability of Batch GD with the speed of SGD; doesn't require the whole dataset in memory at once.
   - Disadvantages: introduces an extra hyperparameter (batch size) to tune.

### 16.9 Autograd (Automatic Differentiation)

- **Backpropagation** is the mathematical algorithm for computing gradients using the chain rule; **Autograd** is the **software system** that automatically performs this differentiation — i.e., Autograd is the implementation that automates the math of backpropagation.
- Rather than manually deriving gradient equations for every operation, Autograd dynamically builds a **computational graph** during the forward pass and automatically computes exact derivatives during the backward pass.
- This automation forms the mathematical core of modern frameworks like PyTorch and TensorFlow — removing human error and enabling rapid experimentation with complex architectures.

---

## 17. Convolutional Neural Networks (CNN)

### 17.1 What is CNN?

- A **Convolutional Neural Network (CNN)**, also called **ConvNet**, is a deep learning algorithm primarily used for **image recognition and classification**.
- It works by extracting features through **convolution** and **pooling** operations, then uses fully connected layers for final classification.
- CNNs excel at identifying patterns/features within images, enabling tasks like image classification, object detection, and other computer vision applications.

### 17.2 Why CNN Instead of a Normal Neural Network?

- A normal (dense/fully connected) network doesn't scale well to images. Example: a 256×256 RGB image has 256×256×3 = 196,608 input neurons — connecting this to just 1,000 hidden neurons requires ~197 million parameters, which is computationally huge and memory-intensive.
- **Images have spatial structure** — a dense network ignores the fact that neighboring pixels are related; it just sees a long flat vector of numbers.
- **Same feature can appear anywhere** in an image (e.g., a cat's eye could be top-left, center, etc.) — a dense network would need to separately learn weights for every possible position, which is very inefficient.

### 17.3 How CNN Solves This

1. **Local Connectivity** — instead of looking at the whole image, a neuron looks only at a small local region (its **receptive field**), since local regions already contain useful information like edges and textures.
2. **Weight Sharing** — the **same filter** (small set of weights) slides across the entire image, drastically reducing the number of parameters compared to a dense network.
3. **Hierarchical Feature Learning** — features are learned layer by layer: early layers detect simple edges → middle layers combine edges into shapes/textures → deeper layers combine shapes into complete objects (e.g., eye → ear → nose → whole face/cat).

### 17.4 CNN Architecture (Typical Flow)

- **Input Image → Convolution Layer → Activation (ReLU) → Pooling Layer → (repeat Conv/ReLU/Pooling) → Flatten → Fully Connected Layer → Output Layer (Softmax)**

### 17.5 The Convolutional Layer

- The **core building block** of a CNN — its purpose is to extract useful features (edges, textures, corners, shapes) from the input image.
- Conceptual process: place a small filter (kernel) on the image → element-wise multiply with the pixels it overlaps → sum the products (+ optional bias) → apply activation (usually ReLU) → slide the filter across the whole image (using a chosen **stride**) → repeat with multiple filters → pass the resulting feature maps to the next layer.

### 17.6 Kernel vs Filter vs Feature Map

- **Kernel** — a small 2D matrix of learnable weights (e.g., 3×3) that slides over a **single channel** of input data, producing one value per position via element-wise multiplication and summation.
- **Filter** — the complete 3D unit tracking a specific feature (like an edge or curve) across the **entire input volume**; composed of one kernel per input channel (e.g., a 3×3 filter on an RGB image actually contains 3 kernels — one per channel — giving a 3×3×3 shape).
- **Feature Map (Activation Map)** — the output produced when a filter is applied across the input image; it highlights *where* a particular feature is present. Each filter produces one feature map; different filters learn different features (edges, curves, textures, shapes).

### 17.7 Stride

- **Stride** = the number of pixels the filter moves across the input image during convolution.
- Stride = 1 → filter moves one pixel at a time (examines every region, more detail, more computation).
- Stride > 1 (e.g., 2) → filter skips pixels, producing a smaller output feature map — faster and more memory-efficient, but may lose some fine detail.

### 17.8 Padding

- **Padding** adds zeros around the border of the input matrix, so the output dimension can be kept equal to the input dimension.
- Preserves information at the borders of the image, which would otherwise be under-represented compared to the center (since border pixels are covered by the filter fewer times without padding).

### 17.9 Pooling Layer

- Goal: **reduce computational load, memory usage, and number of parameters**, while retaining the most important feature information.
- **Max Pooling** — selects the **maximum** value from each region of the feature map; the most commonly used approach, generally gives better results.
- **Average Pooling** — computes the **average** value for each region of the feature map.

### 17.10 Receptive Field

- The **receptive field** is the portion of the input image that influences the output of a particular neuron.
- In the first convolution layer, it's roughly equal to the kernel size (e.g., a 3×3 kernel → 3×3 receptive field).
- As more convolution/pooling layers are stacked, the **effective receptive field grows** — deeper neurons capture larger, more complex structures (e.g., Layer 1 sees an eye → Layer 2 sees eye+nose → Layer 3 sees the entire face).

### 17.11 Fully Connected Layer

- Forms the **last few layers** of a CNN — looks like a regular (dense) neural network, connecting every neuron to every neuron in the next layer.
- Takes the flattened feature vector (from the final convolution/pooling stage) and uses it to classify the image into categories.
- Because all parameters are concentrated here, this layer is especially prone to **overfitting**.

### 17.12 Dropout

- A **regularization** technique where randomly chosen neurons are ignored ("dropped") during training at each stage (commonly a dropout rate of 0.5).
- Reduces node-to-node co-dependency, helping the network learn more robust, generalizable features.
- Most commonly applied after fully connected/dense layers (where overfitting risk is highest).
- **Advantages**: reduces overfitting, improves generalization, makes the model more robust.
- **Disadvantages**: increases training time; too high a dropout rate (e.g., 0.8) can cause **underfitting**.

### 17.13 Softmax Layer

- An activation layer typically applied to the **last layer** of a network, acting as the final classifier.
- Converts the network's raw, non-normalized outputs into a **probability distribution** across classes — all output probabilities sum to 1.0.
- Used for **multi-class classification**; Sigmoid is used instead for binary classification.

### 17.14 Famous CNN Architectures (Brief Overview)

| Architecture | Year | Key Idea |
|---|---|---|
| LeNet-5 | 1998 | Early CNN for handwritten digit recognition |
| AlexNet | 2012 | Deeper/larger version of LeNet; won ILSVRC 2012; first to stack conv layers directly |
| GoogLeNet | 2014 | Introduced the "Inception module"; won ILSVRC 2014 |
| ResNet | 2015 | Introduced **skip connections**, enabling extremely deep networks (152 layers); won ILSVRC 2015 |
| MobileNet | 2017 | Uses depthwise + pointwise convolutions for efficient mobile/edge deployment |
| EfficientNet | 2020 | Uses **compound scaling** (depth, width, resolution together) for better performance with fewer parameters |

---

## 18. Recurrent Neural Networks (RNN)

### 18.1 Why RNN? (Limitation of Feed-Forward Networks)

- A **Feed-Forward Neural Network (FFNN)** has only one route of information flow — straight from input to output — and has **no memory** of previous inputs it has processed.
- This makes FFNNs poor at tasks involving **sequential data** (text, speech, time series), where the output at one step depends on what came before.

### 18.2 What is an RNN?

- A **Recurrent Neural Network (RNN)** is a type of neural network designed to **process sequential data**, where the output from the previous step is fed back as input to the current step.
- Its most important feature is the **Hidden State** (also called **Memory State**) — it remembers information about the sequence seen so far and carries it forward.
- The fundamental processing unit in an RNN is sometimes called a **Recurrent Neuron** / **Recurrent Unit**.

### 18.3 FFNN vs RNN — Comparison Table

| Aspect | FFNN | RNN |
|---|---|---|
| Information flow | One direction, input → output | Cycles via a loop (recurrent connection) |
| Memory | No memory of past inputs | Has memory via hidden state |
| Best suited for | Simple classification/regression | Sequential data (text, speech, time series) |

### 18.4 Architecture of RNN

- An RNN has **3 layers**: Input Layer, Hidden Layer, Output Layer — similar to a feed-forward network, except the hidden layer has **connections pointing backward** (recurrent connections).
- At any given time `t`, the recurrent layer's input is a combination of the **current input x(t)** and the **previous hidden state h(t-1)**.
- The output at any time is used to help improve the network's future output (the hidden state is passed along).

### 18.5 How RNN Works

- Conceptually, at each timestep the RNN applies a **recurrence relation**:

  **hₜ = f(hₜ₋₁, xₜ)**

  - `hₜ` = new hidden state (current), `hₜ₋₁` = old/previous hidden state, `xₜ` = input at current timestep, `f` = function with shared weights.
- The **same function and set of weights are reused at every time step** as the sequence is processed — this weight-sharing across time is what lets RNNs handle sequences of any length.
- "Unfolding" the RNN through time shows it as a chain of repeated units, each passing its hidden state to the next.
- Since RNN works on sequential data, weights are updated using an extended version of backpropagation called **Backpropagation Through Time (BPTT)**.

### 18.6 Activation Functions Used in RNN

- Commonly Sigmoid, Tanh, or ReLU (same functions covered earlier in Section 8) — **Tanh** is the most traditionally used activation inside the RNN's hidden state update, since it keeps values zero-centered between -1 and 1.

### 18.7 Types of RNN

1. **One-to-One** — single input, single output (like a traditional neural network).
2. **One-to-Many** — single input produces multiple outputs over time (e.g., music generation from one seed input).
3. **Many-to-One** — multiple inputs across time steps produce a single output (e.g., sentiment analysis/emotion detection from a sequence of words).
4. **Many-to-Many** — multiple inputs produce multiple outputs (e.g., machine translation — English to French).

### 18.8 Applications of RNN

- **Image Captioning** — generating a caption by analyzing activities/objects present in an image.
- **Time Series Prediction** — e.g., predicting stock prices for a particular month.
- **Natural Language Processing (NLP)** — text mining, sentiment analysis.
- **Machine Translation** — translating input from one language into another (e.g., Google Translate).

---

## 19. Encoder-Decoder (Sequence-to-Sequence) Architecture

### 19.1 What is it?

- The **encoder-decoder architecture** is a widely-used deep learning framework for tasks requiring **sequence-to-sequence (Seq2Seq) modeling** — where input and output sequences may differ in length.
- Common uses: machine translation, text summarization, question-answering systems, chatbots.
- Introduced (in this Seq2Seq form) by Google in 2014.

### 19.2 Why Needed?

- A regular RNN network maps a **fixed-length input to a fixed-length output**, where each input word maps to one output.
- But many real tasks need **variable-length input and output** (e.g., translating a 5-word English sentence into a 7-symbol Chinese sentence) — a plain RNN cannot handle this length mismatch, motivating the encoder-decoder design.

### 19.3 The Two Main Components

1. **Encoder** — takes the input sequence and processes it (typically using an RNN or LSTM) step by step, **summarizing** the entire input into a fixed-size representation called the **"context vector"** (or "thought vector"). It captures the essential information/sequential dependencies of the input.
2. **Decoder** — takes the context vector produced by the encoder as its starting point, and **generates the output sequence** step by step (token by token); it can produce an output sequence of a different length than the input (e.g., a translated sentence, or a chatbot's response).

### 19.4 How It Works (Conceptual)

- Multiple RNN cells are stacked to form the encoder; at every timestep, the hidden state is updated based on the current input and the previous hidden state.
- After all inputs are read, the encoder's **final hidden state** represents the summarized context of the whole input sequence — this is passed to the decoder as its initial state.
- The decoder then generates the output sequence one token at a time, using the context vector and its own previous outputs/hidden state.

### 19.5 Advantages

- **Flexibility with input/output sequences** — handles variable-length input and output, unlike a plain RNN.
- **Versatility in application** — used beyond text, including speech recognition, video captioning, and time series prediction.

### 19.6 Applications

- **Machine Translation** (e.g., Google Translate).
- **Speech Recognition**.
- **Video Captioning** (e.g., generating natural-language descriptions of video content).

---

## 20. Long Short-Term Memory (LSTM)

### 20.1 Drawbacks of RNN (Why LSTM Was Needed)

- A traditional RNN can only remember the **immediate past** — it has a kind of "**short-term memory**" and simply deletes/overwrites the longest-retained information once its limited memory runs out.
- RNNs also suffer from the **Vanishing Gradient Problem** — gradients used to update weights become very small as they're propagated back through many time steps, making it difficult for RNNs to learn **long-term dependencies** in sequences.

### 20.2 What is LSTM?

- **Long Short-Term Memory (LSTM)** is a type of artificial recurrent neural network (RNN) architecture used in deep learning, designed by **Hochreiter & Schmidhuber (1991)** as an improved version of RNN.
- LSTMs address the vanishing gradient problem by introducing a **memory cell** — a container that can hold information for an extended period — along with **specialized gates** that manage the flow of information.
- Great for tasks needing both short-term and long-term memory of sequential data: machine translation, speech recognition, time-series analysis, audio/video classification.

### 20.3 LSTM Architecture — The Three Gates

LSTM architecture is controlled by **three gates**, which decide what information to add to, remove from, and output from the memory cell:

1. **Forget Gate** — decides what information is no longer useful and should be discarded from the cell state (forgets irrelevant information).
2. **Input Gate** — controls what new information is added to/updates the memory cell.
3. **Output Gate** — controls what information from the current cell state is passed on as output (and to the next time step).

- **Cell (Memory Part of LSTM)** — stores the state of a sequence, giving it the ability to keep or forget certain information; this is the "long-term memory" of the network.
- **Hidden State** — acts as the short-term memory of the network, updated at each time step based on the input, the previous hidden state, and the memory cell's current state.

### 20.4 How LSTM Works (Conceptual Role of Each Gate)

- At each timestep, the LSTM receives **three inputs**: the current input, the previous hidden state, and the previous cell state.
- **Forget Gate**: looks at the previous hidden state and current input, and outputs a value between 0 and 1 for each number in the cell state — 0 means "completely forget," 1 means "completely keep."
- **Input Gate**: decides what new information should be stored in the cell state (via a sigmoid layer deciding *which* values to update, combined with a tanh layer creating candidate values to potentially add).
- **Cell State Update**: the old cell state is updated by combining what's kept (via the forget gate) with what's newly added (via the input gate).
- **Output Gate**: decides what part of the (updated) cell state should be output as the new hidden state, passed both to the output layer and to the next time step.

### 20.5 RNN vs LSTM — Comparison Table

| Feature | RNN | LSTM |
|---|---|---|
| Memory | Does not have a dedicated memory unit | Has a special memory cell for long-term dependencies |
| Directionality | Processes sequential data in one direction | Can be trained in both forward and backward directions |
| Training | Easier to train | More difficult to train (due to gates/memory unit complexity) |
| Long-term dependency learning | Limited | Yes (its core strength) |
| Ability to learn sequential data | Yes | Yes |
| Applications | NLP, machine translation, speech/image/video processing | Machine translation, speech recognition, text summarization, NLP, time series forecasting |

### 20.6 Advantages of LSTM

1. **Ability to process sequential data** — well-suited for time series data, natural language, speech recognition, language translation, and sentiment analysis.
2. **Ability to handle long-term dependencies** — specifically designed to address the vanishing gradient problem that occurs in traditional RNNs, making LSTMs well-suited for tasks like predicting stock prices or weather patterns over long sequences.
3. **Memory cell** — allows the network to selectively remember or forget information over long periods, making it more effective at handling complex tasks than plain RNNs.

---

## 21. YOLO (You Only Look Once) — Object Detection

### 21.1 Object Detection — Introduction

- **Object Detection** is a computer vision task that answers **what** is in an image and **where** it is located.
- Combines **Image Classification** (identifying the object) with **Localization** (drawing a bounding box around it).
- Older/legacy models treated detection as a **multi-step pipeline**, making them computationally expensive.

### 21.2 Progression: Classification → Localization → Detection

- **Image Classification** — predicts only the object's category (answers "What is in the image?").
- **Object Localization** — predicts the object's category **and** its bounding box (adds a regression head predicting x, y, w, h — the box center coordinates, width, and height).
- **Object Detection (YOLO)** — extends localization further by detecting **multiple objects simultaneously**, predicting for each: object class, bounding box coordinates, and a confidence score.

### 21.3 What is YOLO?

- **YOLO** = **"You Only Look Once."**
- Core concept: a **single CNN** passes over the full image just **once**.
- Evaluates the **entire image globally** at once, drastically reducing background/context errors compared to models that scan localized windows separately.
- Extremely fast — capable of running at **45+ Frames Per Second (FPS)**, suitable for real-time applications.

### 21.4 Two-Stage vs Single-Stage Detectors

- **Two-Stage Detectors** (e.g., R-CNN, Fast R-CNN, Faster R-CNN) — Step 1: propose regions likely to contain objects; Step 2: classify those proposed regions. Highly accurate, but computationally slow.
- **Single-Stage Detectors** (e.g., YOLO, SSD) — reframe detection as a **single regression problem**, going directly from image pixels to bounding boxes + classes in one pass. Much faster.

### 21.5 How YOLO Works (Pipeline)

- Input Image → Resize → CNN extracts features → Feature map becomes an **S×S grid** (e.g., 7×7) → each grid cell predicts bounding boxes, confidence scores, and class probabilities → compute final scores → **Non-Max Suppression** → final detected objects.

### 21.6 Grid Cells and Predictions

- YOLO divides the image into an **S×S grid** (commonly 7×7 = 49 cells).
- Each grid cell is **responsible for detecting objects whose center falls inside it** — even if the object is larger than the cell.
- Each cell predicts **B bounding boxes** (commonly B=2); each box carries **5 values**: (x, y, w, h, confidence).

### 21.7 What is a Bounding Box?

- `x, y` — the center of the box, expressed as an offset relative to the grid cell (values between 0 and 1).
- `w, h` — width and height, expressed relative to the whole image (values between 0 and 1).

### 21.8 Confidence Score

- Confidence reflects **how sure** the model is that a box contains an object, **and how accurate** that box is.
- Formula (conceptual): **Confidence = P(Object) × IoU**
  - `P(Object)` = 1 if an object's center is in that cell, else 0.
  - `IoU` = how well the predicted box overlaps the true (ground-truth) box.

### 21.9 Intersection over Union (IoU)

- **IoU** measures how well a predicted bounding box overlaps with the ground-truth bounding box.
- Formula (conceptual): **IoU = Area of Overlap / Area of Union**
- A higher IoU means better localization; **IoU ≥ 0.5** is typically considered a correct detection.

### 21.10 Class Probabilities & Combined Score

- Each grid cell also predicts **class probabilities** — the likelihood the detected object belongs to each possible class, **conditioned on an object being present**: P(Class | Object).
- The box's confidence is multiplied by the class probability to get a **class-specific confidence score**: P(Class) × IoU — telling you both "how likely" and "how tight the box is" in one number.

### 21.11 The Full Output Tensor

- For a grid of size S×S, B boxes per cell, and C classes, YOLO's output is a tensor of shape:

  **S × S × (B × 5 + C)**

- Example (YOLOv1 on PASCAL VOC): S=7, B=2, C=20 → **7 × 7 × 30** — 49 cells, each carrying 30 numbers (10 values for two boxes + 20 class probabilities).

### 21.12 Non-Max Suppression (NMS)

- Multiple grid cells may end up detecting the **same object** — NMS removes these duplicate/overlapping boxes, keeping only the best one.
- Algorithm (conceptual steps):
  1. Discard all boxes with a confidence below a threshold (e.g., 0.6).
  2. Pick the box with the **highest confidence score** among the remaining candidates.
  3. Remove/discard any other box that overlaps this chosen box with **IoU above a threshold** (commonly 0.5) — since they likely refer to the same object.
  4. Repeat steps 2–3 until no boxes remain — the boxes kept are the final detections.

### 21.13 The Loss Function (Conceptual)

- YOLO's loss is a **multi-part sum-squared error** with these components:
  1. **Localization Loss** — error in predicted box center (x, y) and size (√w, √h) — square roots are used so errors on small objects aren't overwhelmed by errors on large objects.
  2. **Confidence Loss (object present)** — error in confidence when an object IS present in the cell.
  3. **Confidence Loss (no object)** — error in confidence when no object is present (this term is typically down-weighted, e.g., λ = 0.5, so empty cells don't drown out the signal from cells with real objects).
  4. **Classification Loss** — error in predicted class probabilities, calculated only for grid cells that actually contain an object.
- `λcoord` (commonly 5) boosts the weight given to coordinate accuracy in the total loss.

### 21.14 YOLO Architecture Summary

- YOLO is made up of three key components:
  - **Backbone** — convolutional layers that detect and process key image features (often pre-trained on a classification dataset like ImageNet).
  - **Neck** — uses the backbone's features (with fully connected layers) to make predictions on probabilities and bounding box coordinates.
  - **Head** — the final output layer producing the S×S×(B×5+C) tensor; can be swapped out for transfer learning to other tasks.

### 21.15 Advantages & Limitations of YOLO (v1)

- **Advantages**: extremely fast (real-time detection), single neural network, end-to-end training, sees the whole image (good global context/generalization), suitable for video applications.
- **Limitations**: struggles with small objects; each grid cell can only predict a limited number of objects; lower localization accuracy than some region-based (two-stage) methods; difficulty with objects that are close together.

---

## 22. Concepts of NLP

### 22.1 Linguistics Basics

- **Linguistics** = the systematic study of language — patterns in human language and how it interacts with human behavior, physiology, and culture.
- **Major levels of linguistics**:
  - **Phonetics** — study of individual speech sounds.
  - **Phonology** — study of phonemes (the speech sounds of a specific language).
  - **Morphology** — study of words and meaningful units like prefixes/suffixes.
  - **Syntax** — study of how words are ordered into sentences/phrases.
  - **Semantics** — study of the meaning of sentences.
  - **Pragmatics** — study of how context contributes to meaning.
- **Five structural components of language**:
  - **Phonemes** — smallest unit of sound that can change meaning (no meaning on its own), e.g., changing "bake" to "brake."
  - **Morphemes** — smallest **meaningful** unit of language; can be **free** (stand alone, e.g., "eat") or **bound** (prefixes/suffixes, e.g., "-s", "re-"). Bound morphemes are further split into **derivational** (change meaning/part of speech, e.g., "sad" → "sadness") and **inflectional** (change tense/number, e.g., "cat" → "cats").
  - **Lexemes** — the set of inflected forms of a single word (e.g., "run," "running," "ran" all belong to the lexeme RUN — but "runner" does not, since it's derived).
  - **Syntax** — rules for constructing sentences from words/phrases; word order matters in English ("The baby ate the carrot" ≠ "The carrot ate the baby").
  - **Context** — how tone, body language, and word choice together convey meaning (e.g., "awesome" said sarcastically vs. genuinely).

### 22.2 NLP Pipeline & Preprocessing

- Typical NLP workflow: raw text → preprocessing → feature extraction → modeling → output.
- **NLTK (Natural Language Toolkit)** — the most popular Python library for NLP; supports tokenization, stemming, tagging, parsing, classification, and semantic reasoning. Works with **corpora** (large, structured sets of text) for statistical analysis.

#### Tokenization

- **Tokenization** = the first step of NLP; the process of breaking text into smaller units called **tokens** (also called word segmentation).
- **Bigrams** — tokens of two consecutive words; **Trigrams** — three consecutive words; **N-grams** — any number of consecutive words.
- **Sentence Tokenization** — splitting a paragraph into individual sentences.
- **Common issues in tokenization**: ambiguous abbreviations ("PhD", "Dr."), possessives ("India's"), contractions ("isn't", "I'm"), compound terms ("state-of-the-art"), languages with no spaces (Chinese/Japanese), mixed date/time formats, code-mixed text (chat mixing multiple languages).

#### Text Cleaning / Noise Removal

- **Noise Removal** — removing punctuation, special characters, numeric digits, extra whitespace, and HTML formatting; typically done using regular expressions.
- **Lowercasing** — converting all text to the same case so "Nice", "NICE", and "nice" are treated identically — helps feature extraction techniques like TF-IDF avoid counting the same word multiple times.
- **Removing URLs/HTML tags** — these carry no useful semantic information for most NLP tasks, so they're typically stripped out.
- **Stopwords** — common words (e.g., "a", "an", "the", "is") that carry little useful information for tasks like search or sentiment analysis, and are usually filtered out.

#### Stemming vs Lemmatization

- **Stemming** — reduces a word to its root/base form ("stem") by removing suffixes/prefixes; the resulting stem may **not** be an actual dictionary word (e.g., "historically" → "histori").
  - **Understemming** — two related words that *should* share a stem don't (false negative).
  - **Overstemming** — two unrelated words are incorrectly reduced to the same stem (false positive), e.g., "university" and "universe" → "univers".
  - Common stemmers: **PorterStemmer** (1979, simple and fast, rule-based suffix stripping), **LancasterStemmer** (1990, more aggressive, prone to overstemming), **SnowballStemmer** (an improved "Porter2" stemmer — more accurate).
- **Lemmatization** — reduces a word to its dictionary/canonical form, called the **lemma** (e.g., "runs", "running", "ran" → "run"); unlike stemming, the result is always an actual language word (uses a lexical resource, e.g., WordNet).
- **Key difference**: Stemming is faster but may produce non-words; Lemmatization is slower but always produces valid dictionary words.
- **Applications**: information retrieval/search engines, domain vocabulary analysis, sentiment analysis, document clustering.

### 22.3 Regular Expressions (Regex) — Concept

- A **regular expression** provides a concise way to match patterns of characters/text (a "really smart find/search").
- **Common regex symbols** (recognition-level):

| Symbol | Meaning |
|---|---|
| `^` | Matches the beginning of a line |
| `$` | Matches the end of a line |
| `.` | Matches any character |
| `\s` / `\S` | Whitespace / non-whitespace character |
| `*` | Repeats a character zero or more times |
| `+` | Repeats a character one or more times |
| `[aeiou]` | Matches a single character in the listed set |
| `[^XYZ]` | Matches a single character NOT in the listed set |
| `[a-z0-9]` | A range of characters |
| `( )` | Marks where string extraction starts/ends |

- **Greedy vs Non-Greedy Matching**: by default, repeat characters (`*`, `+`) are **greedy** — they match the largest possible string; adding a `?` (e.g., `+?`) makes them **non-greedy**, matching the smallest possible string instead.
- **Escape character (`\`)**: used to make a special regex character behave as a literal character (e.g., `\$` matches an actual dollar sign).

### 22.4 Text Feature Extraction

#### Bag-of-Words (BoW)

- **Bag-of-Words** converts text into numeric vectors by counting word **frequency**, ignoring grammar and word order — just "how often does each word appear."
- Process: build a vocabulary of all unique words across documents → represent each document as a vector of word counts against that vocabulary.
- **Limitations of BoW**:
  - **Vocabulary** — all words are treated with equal importance; can't tell which words matter more.
  - **Sparsity** — resulting vectors are large and mostly zeros, making them harder to model efficiently.
  - **Meaning** — discards word order, losing context/semantics (e.g., "this is interesting" vs. "is this interesting" would look identical).

#### TF-IDF (Term Frequency–Inverse Document Frequency)

- **TF-IDF** is a statistical measure evaluating how important a word is to a document within a collection/corpus — addresses BoW's "all words matter equally" limitation.
- **Term Frequency (TF)** — how frequently a term occurs in a *specific* document; higher TF = more important within that document.
- **Inverse Document Frequency (IDF)** — measures how *rare* a term is across the whole corpus; higher IDF = rarer (and thus more significant) term.
- **TF-IDF Score = TF × IDF** — a word that's frequent in one document but rare across the corpus gets a high score; common words (like "the") appearing in nearly every document get a low score despite high TF, because their IDF is low.
- **Applications**: search engines (ranking relevance), text summarization (extracting key phrases), document classification (feature selection).

### 22.5 Part-of-Speech (POS) Tagging

- **POS tagging** = assigning a grammatical category label (noun, verb, adjective, etc.) to each token/word in a text, based on its definition and context.
- The **Universal Tagset** (NLTK) has 12 broad tag classes: Verb, Noun, Pronoun, Adjective, Adverb, Adposition, Conjunction, Determiner, Cardinal Number, Particle, Other/Foreign word, Punctuation. (More detailed tagsets like the Penn Treebank break these into finer categories, e.g., NN=noun singular, VBD=verb past tense, JJ=adjective.)
- **Why POS tagging is needed**: helps with **Word Sense Disambiguation** — the same word can have different meanings depending on its POS (e.g., "She saw a **bear**" [noun] vs. "Efforts will **bear** fruit" [verb]).
- **Applications**: Sentiment Analysis (e.g., "like" is only positive when used as a verb, not a conjunction), Text-to-Speech systems, linguistic research, focusing voice-assistant queries (verbs for commands, nouns for topics).

#### POS Tagging Techniques

1. **Rule-Based Tagging** — uses handwritten context rules (e.g., "if an ambiguous word ends in '-ing' and follows a verb, tag it as a verb").
2. **Transformation-Based Tagging** — uses a mix of handcrafted and automatically induced rules generated during training.
3. **Stochastic (Probabilistic) Tagging** — assigns tags based on the probability of a tag sequence; the simplest version uses the most frequent tag seen for a word in training data.

#### Hidden Markov Model (HMM) — Concept

- A **Markov Model** assumes the future state depends only on the **current** state, not on prior history (the "Markov property").
- A **Hidden Markov Model (HMM)** is a statistical Markov model where the actual states (here, POS tags) are **hidden/unobserved** — only some probabilistic function of them (the words) is visible.
- For POS tagging: HMM states represent the POS tags, and the observed outputs represent the words; the tagging task is to find the **most probable tag sequence** for a given sequence of words.
- Common issue: **sparse data** — some tag sequences may never appear in a limited training corpus; **smoothing techniques** are used to avoid assigning zero probability to unseen sequences.

### 22.6 Parsing

#### Shallow Parsing (Chunking)

- **Parsing** = determining the syntactic structure of a sentence by analyzing its words according to grammar rules.
- **Shallow Parsing** (a.k.a. **chunking** or "light parsing") — extracts only a **limited part** of the syntactic structure, grouping tokens into higher-level phrases without full sentence structure.
- **Chunk types**: Noun Phrase (NP), Verb Phrase (VP), Prepositional Phrase (PP).
- Example: POS tagging output → `My/PRP$ dog/NN likes/VBZ his/PRP$ food/NN`; Chunking output → `[NP My Dog] [VP likes] [NP his food]`.
- A **POS tagger** can be thought of as returning only the bottom-most tier of a full parse tree — fast, but with limited information; a **shallow parser/chunker** sits in between a plain POS tagger and a full parser.

#### Deep Parsing (Full Parsing)

- **Deep Parsing** gives a **complete syntactic structure** to a sentence, typically using grammar formalisms like Context-Free Grammar (CFG) or Probabilistic CFG (PCFG).
- **Constituency Parsing** — represents a sentence's internal structure hierarchically using **phrase structure rules** (e.g., **S → NP VP**, meaning a sentence = Noun Phrase + Verb Phrase); builds a tree with words as leaves and phrase tags (NP, VP, PP) as internal nodes.
- **Dependency Parsing** — represents relationships as **directed links (dependencies)** directly between individual words, rather than building up phrases; the word with no dependency (usually the main verb) is the **root** of the sentence, and all other words connect to it directly or indirectly. Example dependency tags: `det` (determiner relationship), `amod` (adjectival modifier), `nsubj` (subject of a clause).
- **Constituency vs Dependency**: constituency parsing focuses on relationships between phrases; dependency parsing focuses directly on relationships between individual words.

#### Shallow vs Deep Parsing — Comparison Table

| Aspect | Shallow Parsing | Deep Parsing |
|---|---|---|
| Structure given | Limited/partial syntactic info | Complete syntactic structure |
| Also known as | Chunking | Full parsing |
| Best suited for | Information extraction, text mining | Complex NLP apps: dialogue systems, summarization |
| Speed | Fast | Slower |

#### Named Entity Recognition (NER)

- **NER** (also called entity identification/extraction) automatically identifies **named entities** in text and classifies them into predefined categories — e.g., names of people, organizations, locations, dates, monetary values, percentages.

### 22.7 Text Summarization

- **Text Summarization** = creating a short, coherent, fluent summary of a longer document without losing vital/key information.
- **Extractive Summarization** — selects and combines a subset of the most important existing sentences/words directly from the source text (like a "highlighter").
- **Abstractive Summarization** — uses deep learning to generate new, paraphrased sentences that capture the meaning, similar to how a human would summarize (like a "pen" — can produce sentences not present in the original text).

---
**End of core NLP (Deep Learning syllabus) content.** Transformer-based NLP topics — Attention Mechanisms, Encoder-Decoder Transformers, BERT, GPT, LLMs, RAG, Agentic AI — belong to the separate **Generative AI** notes and are intentionally NOT included here.

---

## 23. Additional Topics — Not Covered in Teacher's Class Materials
*(These topics are in your syllabus but were not present in any uploaded PPT/asset. Added from general knowledge to complete syllabus coverage.)*

### 23.1 Deep Feedforward Networks (DFN)

- Also called **Multilayer Perceptrons (MLPs)** — the foundational architecture of deep learning.
- Goal: approximate some target function f\*, by defining a mapping **y = f(x; θ)** and learning the parameters **θ** that best fit that function.
- Called "**feedforward**" because information flows in **one direction only** — input → hidden layer(s) → output — with **no feedback/recurrent connections** looping back (unlike RNNs).
- Serves as the conceptual foundation on which more specialized architectures (CNNs, RNNs) add extra structure (e.g., convolution, recurrence).

### 23.2 Regularization for Deep Learning

- Techniques used to reduce **overfitting** in deep neural networks (beyond just Dropout, already covered under CNN).
- **L1/L2 Regularization (Weight Decay)** — adds a penalty on large weight values to the loss function, discouraging overly complex models (same L1/L2 concept as in ML, applied to neural network weights).
- **Dropout** — randomly deactivates neurons during training (already detailed in Section 17.12) — a regularization technique originally popularized in CNNs but used broadly across deep learning.
- **Data Augmentation** — artificially increases the size/diversity of training data by applying transformations (rotation, flipping, cropping, zoom for images) — helps the model generalize better without collecting new data.
- **Early Stopping** — halts training once validation performance stops improving, preventing the model from continuing to overfit the training data.
- **Batch Normalization** — normalizes the inputs to each layer during training (stabilizing their mean/variance), which speeds up training, allows higher learning rates, and provides a mild regularization effect by reducing "internal covariate shift."

### 23.3 Autoencoders

- An **Autoencoder** is an **unsupervised** neural network trained to reconstruct its own input as closely as possible.
- **Architecture**: an **Encoder** compresses the input into a smaller, lower-dimensional representation (called the **latent space** or **bottleneck**), and a **Decoder** reconstructs the original input from that compressed representation.
- Trained by minimizing **reconstruction loss** — the difference between the original input and the reconstructed output.
- **Types of Autoencoders**:
  - **Vanilla Autoencoder** — basic encoder-decoder reconstruction.
  - **Denoising Autoencoder** — trained to reconstruct a clean input from a deliberately corrupted/noisy version, making it robust to noise.
  - **Sparse Autoencoder** — adds a sparsity constraint on the hidden layer, forcing the network to learn more meaningful, compact features.
  - **Variational Autoencoder (VAE)** — a generative version that learns a probability distribution over the latent space, allowing it to generate new, similar data samples (not just reconstruct existing ones).
- **Applications**: dimensionality reduction, image denoising, anomaly detection (poor reconstruction = likely anomaly), image compression, and as a building block for generative models.

### 23.4 Object Segmentation and Tracking

- **Object Segmentation** goes beyond Object Detection (bounding boxes) to classify images at the **pixel level**.
  - **Semantic Segmentation** — classifies every pixel into a category (e.g., "road," "car," "person"), but does **not** distinguish between separate instances of the same class.
  - **Instance Segmentation** — distinguishes between individual **instances** of objects (e.g., two different people each get their own separate mask), combining detection with pixel-level masks (e.g., Mask R-CNN).
- **Object Tracking** — follows a detected object across **consecutive video frames over time**, maintaining a consistent identity for that object as it moves.
  - Common approaches combine object detection with tracking algorithms (e.g., **DeepSORT**, which pairs a detector with a tracking algorithm to maintain object identities across frames).

### 23.5 Introduction to MLOps

- **MLOps (Machine Learning Operations)** = a set of practices combining Machine Learning, DevOps, and Data Engineering to reliably build, deploy, and maintain ML models in production.
- **Key components**:
  - **Version Control** — tracking versions of data, code, and models (not just code, unlike traditional software).
  - **CI/CD Pipelines for ML** — automating the testing, integration, and deployment of ML models.
  - **Model Monitoring** — tracking a deployed model's performance over time and detecting **data/model drift** (when real-world data starts to differ from training data, degrading performance).
  - **Automated Retraining Pipelines** — automatically retraining models as new data becomes available.
  - **Model Registry** — a centralized system for storing, versioning, and managing trained models.
- **Goals**: reproducibility, scalability, collaboration between data scientists and engineers, and faster/more reliable deployment cycles.
- **Common tools** (recognition-level): MLflow (experiment tracking/model registry), Kubeflow, Docker/Kubernetes (containerization and orchestration for deployment).

---
*(This table will be updated every time a new architecture/model topic is covered — lists the model and its category.)*
---

## Architectures/Models Studied So Far
*(This table will be updated every time a new architecture/model topic is covered — lists the model and its category.)*

| Architecture/Model | Category |
|---|---|
| Convolutional Neural Network (CNN) | Computer Vision — Image Classification/Feature Extraction |
| LeNet-5 | CNN Architecture — Handwritten Digit Recognition |
| AlexNet | CNN Architecture — Image Classification |
| GoogLeNet | CNN Architecture — Image Classification (Inception module) |
| ResNet | CNN Architecture — Image Classification (Skip connections) |
| MobileNet | CNN Architecture — Efficient/Mobile Vision |
| EfficientNet | CNN Architecture — Compound-Scaled Image Classification |
| Recurrent Neural Network (RNN) | Deep Learning — Sequence Modeling |
| Encoder-Decoder (Seq2Seq) | Deep Learning — Sequence-to-Sequence / NLP |
| Long Short-Term Memory (LSTM) | Deep Learning — Sequence Modeling (Long-term dependencies) |
| YOLO (You Only Look Once) | Computer Vision — Real-time Object Detection |
| Hidden Markov Model (HMM) | NLP — Probabilistic Sequence Tagging (POS Tagging) |
| Bag-of-Words | NLP — Text Feature Extraction |
| TF-IDF | NLP — Text Feature Extraction |
| Autoencoder (Vanilla/Denoising/Sparse/VAE) | Unsupervised Deep Learning — Reconstruction/Generative |

---

## Library & Import Cheat Sheet
*(to be filled in as frameworks/layers/functions are covered)*

| Component/Technique | Library | Import Statement |
|---|---|---|
| TensorFlow | tensorflow | `import tensorflow as tf` |
| Keras (high-level API, part of TensorFlow) | tensorflow | `from tensorflow import keras` |
| PyTorch | torch | `import torch` |
| Optimizers (Adam, SGD, RMSprop) | tensorflow.keras | `from tensorflow.keras.optimizers import Adam, SGD, RMSprop` |
| CNN Layers (Conv2D, MaxPooling2D, Flatten, Dropout, Dense) | tensorflow.keras | `from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense` |
| RNN Layer | tensorflow.keras | `from tensorflow.keras.layers import SimpleRNN` |
| LSTM Layer | tensorflow.keras | `from tensorflow.keras.layers import LSTM` |
| GRU Layer (used in some Encoder-Decoder models) | tensorflow.keras | `from tensorflow.keras.layers import GRU` |
| YOLO (pre-trained object detection) | ultralytics | `from ultralytics import YOLO` |
| Word/Sentence Tokenization | nltk | `from nltk.tokenize import word_tokenize, sent_tokenize` |
| Stemming (Porter/Lancaster/Snowball) | nltk | `from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer` |
| Lemmatization | nltk | `from nltk.stem import WordNetLemmatizer` |
| POS Tagging | nltk | `from nltk import pos_tag` |
| Regular Expressions | re (built-in) | `import re` |
| Bag-of-Words | scikit-learn | `from sklearn.feature_extraction.text import CountVectorizer` |
| TF-IDF | scikit-learn | `from sklearn.feature_extraction.text import TfidfVectorizer` |
| Batch Normalization | tensorflow.keras | `from tensorflow.keras.layers import BatchNormalization` |
| Data Augmentation (images) | tensorflow.keras | `from tensorflow.keras.preprocessing.image import ImageDataGenerator` |
| Early Stopping | tensorflow.keras | `from tensorflow.keras.callbacks import EarlyStopping` |
| MLOps Experiment Tracking | mlflow | `import mlflow` |
