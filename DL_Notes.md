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

- The simplest type of neural network — a single neuron that takes inputs, multiplies them by weights, adds a bias, and passes the result through an activation function to give an output (usually 0 or 1).
- It's the basic building block of neural networks, mainly used for simple binary classification.

### 6.2 How a Perceptron Works

- Three steps:
  1. Takes inputs (x₁, x₂, ... xₙ) — features/signals.
  2. Multiplies each by a weight (w₁, w₂, ... wₙ) — weights represent importance.
  3. Adds a bias, then applies an activation function to decide the output.
- Core formula (conceptual):

  **ŷ = f(Σ wᵢxᵢ + b)**

  - `xᵢ` = inputs, `wᵢ` = weights, `b` = bias, `f` = activation function, `ŷ` = output/prediction.

### 6.3 Types of Perceptron

- **Single-Layer Perceptron** — has only one layer of weights connecting input directly to output. Can only solve linearly separable problems (e.g., AND, OR gates) — can't solve problems like XOR.
- **Multi-Layer Perceptron (MLP)** — has one or more hidden layers between input and output. Can solve complex, non-linear problems (like XOR) — this is basically what we call a Deep Feedforward Network.

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

| Term | Simple Explanation |
|---|---|
| **What is an Activation Function?** | A mathematical function applied to a neuron's output that decides whether it should "fire" and what value to pass forward. |
| **Why is an Activation Function Required?** | Without it, the whole network becomes just a linear equation, no matter how many layers — it couldn't learn complex patterns. Activation functions add non-linearity, letting the network learn complex relationships. |
| **Sigmoid (Logistic) Function** | Squashes output between 0 and 1. Good for probabilities, but suffers from vanishing gradient for very high/low inputs. Formula: 1 / (1 + e^-x) |
| **Tanh (Hyperbolic Tangent) Function** | Squashes output between -1 and 1. Centered at 0 (better than sigmoid), but still has vanishing gradient issue for extreme values. |
| **ReLU (Rectified Linear Unit)** | Outputs the input if positive, else 0. Formula: max(0, x). Fast, simple, most widely used — avoids vanishing gradient for positive values. |
| **The Dying ReLU Problem** | If a neuron's input is always negative, ReLU always outputs 0 — the neuron "dies" and stops learning permanently. |
| **Leaky ReLU & ELU** | • **Leaky ReLU** — allows a small negative slope instead of 0 for negative inputs (fixes dying ReLU).<br>• **ELU (Exponential Linear Unit)** — uses a smooth exponential curve for negative values — helps keep average output closer to 0, better learning in some cases. |
| **Vanishing Gradient Problem** | When gradients become very small during backpropagation in deep networks (common with sigmoid/tanh), causing early layers to learn very slowly or not at all. |

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

| Aspect | TensorFlow | PyTorch |
|---|---|---|
| Developed by | Google | Meta (Facebook) |
| Ease of use | Steeper learning curve (better now with Keras) | More beginner-friendly, Pythonic |
| Graph type | Static graph (older) / now supports dynamic too | Dynamic graph (define-as-you-go) — easier debugging |
| Debugging | Harder (especially older versions) | Easier — feels like normal Python code |
| Deployment | Strong (TensorFlow Serving, TF Lite for mobile) | Improving (TorchServe, ONNX) |
| Popularity | Widely used in industry/production | Widely used in research/academia |
| Community | Large, mature | Large, growing fast |
| Visualization | TensorBoard | TensorBoard support + others |

**In short:**
- TensorFlow → better for production/deployment.
- PyTorch → better for research/experimentation, easier to learn and debug.

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

| Term | Simple Explanation |
|---|---|
| **What is Optimization in Deep Learning?** | The process of adjusting model weights to reduce error and improve predictions. Goal: find weights that give the lowest possible loss. |
| **Cost Function** | Measures how wrong the model's predictions are compared to actual values. Also called loss function. Lower cost = better model. |
| **What is Gradient Descent?** | An algorithm that adjusts weights step-by-step in the direction that reduces the cost function, using the gradient (slope) of the error. |
| **Backpropagation** | The method used to calculate gradients — it sends the error backward through the network (output → input), telling each weight how much it contributed to the mistake. |
| **The Vanishing Gradient Problem** | In deep networks, gradients become very small as they move backward through layers. This makes early layers learn very slowly or stop learning altogether. |
| **Why ReLU Fixes the Vanishing Gradient Problem** | ReLU outputs the input directly if positive (no squishing like sigmoid/tanh), so gradients stay larger and don't shrink to near-zero across layers. |
| **Types of Gradient Descent** | • **Batch Gradient Descent** — uses entire dataset per update. Slow but stable.<br>• **Stochastic Gradient Descent (SGD)** — uses one sample per update. Fast but noisy.<br>• **Mini-Batch Gradient Descent** — uses small batches. Best balance — most commonly used. |
| **Autograd (Automatic Differentiation)** | A feature in frameworks (like PyTorch) that automatically calculates gradients for you — no need to manually compute derivatives, making backpropagation easy to implement. |

---

## 17. Convolutional Neural Networks (CNN)

| Term | Simple Explanation |
|---|---|
| **CNN** | A smart program that looks at images and learns to recognize things in them, like a cat or a car. |
| **Convolutional Layer** | Like sliding a small magnifying glass over the picture to spot lines, edges, and shapes. |
| **Kernel vs Filter vs Feature Map** | • **Kernel** — the tiny magnifying glass itself.<br>• **Filter** — one full "tool" made of kernels, used to find one specific thing (like edges).<br>• **Feature Map** — the result you get after using that tool on the image. |
| **Stride** | How big a jump the magnifying glass takes each time it moves. Bigger jump = faster but rougher look. |
| **Padding** | Adding a blank border around the image so the edges don't get ignored. |
| **Receptive Field** | How much of the original image one point in the result "can see." Gets bigger as you go deeper. |
| **Pooling Layer** | Shrinking the picture info by keeping only the most important parts. Like summarizing a paragraph into one sentence. |
| **Dropout** | Randomly switching off some parts of the network while learning, so it doesn't get lazy or memorize too much. |
| **Fully Connected Layer** | Takes all the learned info and connects it together to make a final decision. |
| **Softmax Layer** | Turns the final scores into percentages, like "90% cat, 10% dog." |

**Overall Flow:** Image → Spot Patterns (Conv) → Shrink (Pooling) → Repeat → Combine Info → Decide → Give Percentages (Softmax)

---

## 18. Recurrent Neural Networks (RNN)

| Term | Simple Explanation |
|---|---|
| **Why RNN? (Limitation of FFNN)** | Feed-Forward Neural Networks (FFNN) treat every input as separate — no memory of what came before. So they can't handle sequences like sentences or time-series data, where order and past info matter. |
| **What is an RNN?** | A neural network with "memory" — it remembers previous inputs while processing new ones. Good for sequential data like text, speech, or time-series. |
| **FFNN vs RNN** | • **FFNN** — data flows one way, no memory, good for fixed-size inputs (images).<br>• **RNN** — has loops, remembers past info, good for sequences (text, speech, time-series). |
| **Architecture of RNN** | Same network repeats itself for each step in the sequence, passing a "hidden state" (memory) from one step to the next. |
| **Working of RNN** | 1. Takes current input + previous hidden state.<br>2. Combines them to create a new hidden state.<br>3. Passes this hidden state to the next step and also gives an output.<br>4. Repeats for the whole sequence. |
| **Activation Functions Used in RNN** | • **Tanh** (most common) — keeps values between -1 and 1.<br>• **Sigmoid** — used in gates (like in LSTM/GRU).<br>• **ReLU** — sometimes used, but can cause instability in RNNs. |
| **Types of RNN** | • **One-to-One** — single input → single output (like normal NN).<br>• **One-to-Many** — single input → sequence output (e.g., image captioning).<br>• **Many-to-One** — sequence input → single output (e.g., sentiment analysis).<br>• **Many-to-Many** — sequence input → sequence output (e.g., translation). |
| **Applications of RNN** | Language translation, speech recognition, text generation, sentiment analysis, time-series prediction (stocks, weather). |

---

## 19. Encoder-Decoder (Sequence-to-Sequence) Architecture

| Term | Simple Explanation |
|---|---|
| **What is it?** | A model made of two parts — an Encoder and a Decoder — used to convert one sequence into another sequence (e.g., English sentence → French sentence). |
| **Why Needed?** | Normal RNNs struggle when input and output sequences have different lengths (like translating a 5-word sentence into a 7-word sentence). Encoder-Decoder handles this by separating "understanding" and "generating." |
| **The Two Main Components** | • **Encoder** — reads the input sequence and compresses it into a fixed-size summary (called context vector).<br>• **Decoder** — takes that summary and generates the output sequence step by step. |
| **How It Works (Conceptual)** | 1. Encoder processes input word-by-word, updating its hidden state.<br>2. Final hidden state = context vector (summary of whole input).<br>3. Decoder takes this context vector as starting point.<br>4. Decoder generates output one word at a time, using previous output + hidden state. |
| **Advantages** | • Handles variable-length input and output.<br>• Works well for translation, summarization, and similar tasks.<br>• Flexible — can be combined with attention for better performance. |
| **Applications** | Machine translation (English → French), text summarization, chatbots, speech-to-text, image captioning. |

---

## 20. Long Short-Term Memory (LSTM)

| Term | Simple Explanation |
|---|---|
| **Drawbacks of RNN (Why LSTM Was Needed)** | RNNs forget long-past information — this is called the "vanishing gradient" problem. Over long sequences, older info fades away, so RNNs struggle with long-term memory. |
| **What is LSTM?** | A special type of RNN designed to remember information for longer periods. It uses "gates" to control what to keep, forget, or output. |
| **LSTM Architecture — The Three Gates** | • **Forget Gate** — decides what old info to throw away.<br>• **Input Gate** — decides what new info to add to memory.<br>• **Output Gate** — decides what info to pass to the next step/output. |
| **How LSTM Works** | 1. Forget Gate looks at current input + previous memory, decides what to discard.<br>2. Input Gate decides what new info is important and adds it to memory (cell state).<br>3. Cell state gets updated (old memory − forgotten parts + new info).<br>4. Output Gate decides what part of updated memory to output as hidden state.<br>5. This hidden state + cell state move to the next time step. |
| **Advantages of LSTM** | • Remembers long-term dependencies better than RNN.<br>• Solves vanishing gradient problem.<br>• Good for long sequences (long texts, long time-series).<br>• Widely used in translation, speech recognition, text generation. |

---

## 21. YOLO (You Only Look Once) — Object Detection

| Term | Simple Explanation |
|---|---|
| **Object Detection** | Finding objects in an image AND drawing boxes around their location (unlike classification, which just names the object). |
| **Progression: Classification → Localization → Detection** | • **Classification** — "What is in the image?" (one label)<br>• **Localization** — "Where is that one object?" (one box + label)<br>• **Detection** — "What objects are where?" (multiple boxes + labels) |
| **What is YOLO?** | A fast object detection algorithm that looks at the image only once and predicts all objects + their locations in a single pass. |
| **Two-Stage vs Single-Stage Detectors** | • **Two-Stage** (e.g., R-CNN) — first proposes regions, then classifies them. Accurate but slow.<br>• **Single-Stage** (e.g., YOLO) — predicts boxes and classes directly in one shot. Faster, slightly less accurate (in early versions). |
| **How YOLO Works (Pipeline)** | 1. Image divided into a grid.<br>2. Each grid cell predicts bounding boxes + confidence + class probabilities.<br>3. Combine box confidence and class probability.<br>4. Remove overlapping/weak boxes using NMS.<br>5. Final boxes = detected objects. |
| **Grid Cells and Predictions** | Image split into an S×S grid. Each cell is responsible for detecting objects whose center falls inside it. |
| **What is a Bounding Box?** | A rectangle (x, y, width, height) showing where an object is located in the image. |
| **Confidence Score** | Tells how sure the model is that a box contains an object AND how accurate that box is. |
| **Intersection over Union (IoU)** | Measures overlap between predicted box and actual (ground truth) box. Higher IoU = better prediction. IoU = Overlap Area / Total Combined Area |
| **Class Probabilities & Combined Score** | Each cell also predicts probability of each class (e.g., dog, car). Final score = confidence score × class probability. |
| **The Full Output Tensor** | Each grid cell outputs: [box coordinates, confidence score, class probabilities] — combined into one big tensor for the whole image. |
| **Non-Max Suppression (NMS)** | Removes duplicate/overlapping boxes for the same object, keeping only the one with the highest confidence. |
| **The Loss Function (Conceptual)** | Combines errors from: box position, box size, confidence score, and class prediction — all added together to train the model. |
| **YOLO Architecture Summary** | A CNN backbone extracts features → final layers predict grid-based boxes, confidence, and class scores — all in one forward pass. |
| **Advantages & Limitations of YOLO (v1)** | • **Advantages** — Very fast, good for real-time detection, sees whole image (less background error).<br>• **Limitations** — Struggles with small/overlapping objects, less accurate than two-stage detectors, limited boxes per grid cell. |

---

## 22. Concepts of NLP

| Term | Simple Explanation |
|---|---|
| **Concepts of NLP** | NLP (Natural Language Processing) = teaching computers to understand, interpret, and generate human language (text/speech). |
| **Linguistics Basics** | Study of language structure — includes:<br>• **Phonetics** — sounds<br>• **Morphology** — word forms<br>• **Syntax** — sentence structure/grammar<br>• **Semantics** — meaning<br>• **Pragmatics** — context-based meaning |
| **NLP Pipeline & Preprocessing** | Steps to prepare raw text for a model: Text → Cleaning → Tokenization → Stopword Removal → Stemming/Lemmatization → Feature Extraction → Model |
| **Tokenization** | Breaking text into smaller units — words or sentences. E.g., "I love NLP" → ["I", "love", "NLP"] |
| **Text Cleaning / Noise Removal** | Removing unwanted stuff — punctuation, special characters, extra spaces, stopwords (like "the", "is"), HTML tags, etc. |
| **Stemming vs Lemmatization** | • **Stemming** — chops word to root form, often crude (e.g., "studies" → "studi").<br>• **Lemmatization** — converts word to proper dictionary root using grammar rules (e.g., "studies" → "study"). |
| **Regular Expressions (Regex) — Concept** | Pattern-matching tool to find/extract specific text patterns (e.g., emails, dates, phone numbers) using symbols/rules. |
| **Text Feature Extraction** | Converting text into numbers so models can understand it. Common methods:<br>• Bag of Words (BoW)<br>• TF-IDF<br>• Word Embeddings (Word2Vec, GloVe) |
| **Part-of-Speech (POS) Tagging** | Labeling each word with its grammatical role — noun, verb, adjective, etc. E.g., "Dog runs fast" → Dog(Noun) runs(Verb) fast(Adverb) |
| **Parsing** | Analyzing grammatical structure of a sentence — shows relationships between words (like a tree structure). |
| **Text Summarization** | Automatically creating a short version of a longer text while keeping key meaning. Two types:<br>• **Extractive** — picks important sentences directly from text.<br>• **Abstractive** — generates new sentences that capture the meaning. |

---
**End of core NLP (Deep Learning syllabus) content.** Transformer-based NLP topics — Attention Mechanisms, Encoder-Decoder Transformers, BERT, GPT, LLMs, RAG, Agentic AI — belong to the separate **Generative AI** notes and are intentionally NOT included here.

---

## 23. Additional Topics — Not Covered in Teacher's Class Materials
*(These topics are in your syllabus but were not present in any uploaded PPT/asset. Added from general knowledge to complete syllabus coverage.)*

| Term | Simple Explanation |
|---|---|
| **Deep Feedforward Networks (DFN)** | Basic neural networks where data flows only forward — input → hidden layers → output, no loops or memory. Foundation of most deep learning models (also called Multi-Layer Perceptrons). |
| **Regularization for Deep Learning** | Techniques to prevent overfitting (model memorizing training data instead of learning patterns). Common methods:<br>• **L1/L2 Regularization** — penalize large weights.<br>• **Dropout** — randomly turn off neurons during training.<br>• **Early Stopping** — stop training when validation error starts increasing.<br>• **Data Augmentation** — create more training data variations. |
| **Autoencoders** | Neural networks that learn to compress data and then reconstruct it back. Made of:<br>• **Encoder** — compresses input into smaller representation.<br>• **Decoder** — rebuilds original input from that representation.<br>Used for dimensionality reduction, denoising, and anomaly detection. |
| **Object Segmentation and Tracking** | • **Segmentation** — dividing an image into meaningful parts/pixels (e.g., separating a person from background).<br>&nbsp;&nbsp;- **Semantic Segmentation** — labels each pixel by class (no separate objects).<br>&nbsp;&nbsp;- **Instance Segmentation** — labels each object separately (even same class).<br>• **Tracking** — following a detected object across video frames over time (e.g., tracking a car in a video). |
| **Introduction to MLOps** | Practices to manage the full ML lifecycle — from building to deploying and maintaining models in production. Includes:<br>• Version control (data, code, models)<br>• CI/CD pipelines for ML<br>• Model monitoring & retraining<br>• Automation of training/deployment<br>Goal: make ML models reliable, scalable, and easy to update in real-world use. |

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
