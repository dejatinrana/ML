# Machine Learning — Exam Revision Notes

---

## 1. Introduction to Machine Learning

### 1.1 Definition of Machine Learning

- ML = subfield of AI where systems learn patterns from data to improve performance without explicit rule-based programming.
- Tom Mitchell's definition: a program learns from experience **E** w.r.t. task **T** and performance measure **P**, if performance on **T** (measured by **P**) improves with **E**.
- ML = Data + Algorithm → Model that predicts on new/unseen data.

### 1.2 Why is ML Needed?

- Traditional programming fails when rules are unknown, too many, or keep changing.
- Needed to automate decisions, find hidden patterns, adapt with new data, and handle large/unstructured data (images, text, audio).

### 1.3 Advantages of ML

- Improves automatically with more data.
- Handles large, complex datasets.
- Finds patterns humans may miss.
- Works across many domains.
- Reduces manual intervention.
- Enables personalization.

### 1.4 Disadvantages of ML

- Needs large, good-quality data.
- Can inherit bias from training data.
- Often a "black box" (low interpretability).
- Computationally expensive.
- Risk of overfitting/underfitting.
- Needs expertise and tuning.
- Performance can degrade over time (data drift).

### 1.5 Is Machine Learning a Language?

- **No.**
- ML is a field/discipline of AI, not a language.
- Implemented using languages like Python/R, but ML itself is a concept/methodology.

---

## 2. Types of Machine Learning

### 2.1 Four Core Types

1. **Supervised** — labeled data; learns input→output mapping (e.g., spam detection).
2. **Unsupervised** — unlabeled data; finds hidden structure/groups (e.g., clustering).
3. **Semi-Supervised** — small labeled + large unlabeled data (e.g., medical imaging).
4. **Reinforcement** — agent learns via rewards/penalties from environment (e.g., game AI).

### 2.2 Other Types

- **Self-Supervised** — labels generated from data itself (used in NLP/LLMs).
- **Online Learning** — model updates incrementally as new data arrives.
- **Batch Learning** — trained once on full dataset; no updates until retrained.
- **Transfer Learning** — reuses knowledge from one task for another related task.

### 2.3 Comparison Table

| Type | Data | Goal | Example |
|---|---|---|---|
| Supervised | Labeled | Input→output mapping | Spam detection |
| Unsupervised | Unlabeled | Find hidden patterns | Customer segmentation |
| Semi-supervised | Small labeled + large unlabeled | Generalize from few labels | Medical imaging |
| Reinforcement | Rewards from environment | Learn optimal policy | Game-playing agents |

---

## 3. AIML — What & Why Used Together

- **AIML** = combined term for **Artificial Intelligence + Machine Learning**, since ML is a core subset/technique used to build AI systems.
- Used together because:
  - AI is the broader goal (machines mimicking human intelligence).
  - ML is the primary method used today to achieve that goal (learning from data instead of hardcoded rules).
  - Most modern "AI" systems in practice are actually ML models — so the terms are bundled in industry/job titles and courses.

---

## 4. What is Analysis? Types of Analysis

- **Analysis** = process of examining data to extract meaningful insights, patterns, or conclusions.

### 5.1 Main Four Types

1. **Descriptive Analysis** — summarizes *what happened* (e.g., total sales last month).
2. **Diagnostic Analysis** — explains *why it happened* (e.g., why sales dropped).
3. **Predictive Analysis** — forecasts *what will happen* (uses ML models).
4. **Prescriptive Analysis** — recommends *what action to take* next.

### 5.2 Additional Types

5. **Exploratory Data Analysis (EDA)** — investigates data to discover patterns, spot anomalies, test hypotheses, and check assumptions, usually via visualizations/summary stats, before modeling.
6. **Inferential Analysis** — uses a data sample to draw conclusions/generalizations about a larger population, often using statistical tests.

### 5.3 Comparison Table

| Type | Question Answered |
|---|---|
| Descriptive | What happened? |
| Diagnostic | Why did it happen? |
| Predictive | What will happen? |
| Prescriptive | What should we do? |
| EDA | What patterns exist in the data? |
| Inferential | What can we conclude about the population from a sample? |

---

## 5. Rough Idea of Working of ML (High-Level)

- Simple end-to-end pipeline (conceptual, no deep math):
  1. **Collect data** relevant to the problem.
  2. **Preprocess data** (clean, handle missing values, encode, scale).
  3. **Split data** into training and testing sets.
  4. **Choose a model/algorithm** suited to the problem.
  5. **Train the model** on training data (model learns patterns).
  6. **Evaluate the model** on test data (accuracy, error metrics).
  7. **Tune hyperparameters** to improve performance.
  8. **Deploy the model** for real-world predictions.
- Core idea: model finds a mathematical relationship between input features and output, then applies that relationship to new data.

---

## 6. Data Processing

### 7.1 Common Preprocessing Steps

- **Data Cleaning** — handle missing values, remove duplicates, fix inconsistent entries.
- **Handling Outliers** — detect and treat extreme values that can skew the model.
- **Feature Encoding** — convert categorical data into numerical form (e.g., One-Hot Encoding, Label Encoding).
- **Feature Scaling** — bring numerical features to a common scale (Standardization, Normalization, etc.) — covered in detail below.
- **Feature Selection/Extraction** — keep only relevant features or reduce dimensionality (e.g., PCA).
- **Data Splitting** — divide into training, validation, and test sets.

### 7.2 Feature Scaling

- Feature Scaling = technique to bring all numerical features onto a similar scale/range, so no single feature dominates the model due to larger magnitude.
- Needed because algorithms like KNN, SVM, Gradient Descent-based models are sensitive to feature magnitude.

#### 7.2.1 Standardization (Z-Score Scaling)

- Rescales data so it has **mean = 0** and **standard deviation = 1**.
- Formula:

  **z = (x − μ) / σ**

  - `x` = original data value
  - `μ` (mu) = mean of the feature
  - `σ` (sigma) = standard deviation of the feature
  - `z` = standardized value

- Does not bound values to a fixed range (unlike normalization) — values can be negative or beyond ±1.
- Useful when data follows a roughly normal (Gaussian) distribution, or for algorithms assuming zero-centered data (e.g., PCA, SVM, Logistic Regression, Gradient Descent-based models).

#### 7.2.2 Normalization (Min-Max Scaling)

- Rescales data into a **fixed range**, usually **[0, 1]**.
- Formula:

  **x' = (x − min) / (max − min)**

  - `x` = original value
  - `min`, `max` = minimum and maximum of the feature
  - `x'` = normalized value (between 0 and 1)

- Unlike Standardization, output is bounded to a fixed range — sensitive to outliers (a single extreme value shifts min/max for everyone).
- Useful for algorithms that need bounded input, e.g., Neural Networks, KNN, image pixel data.

#### 7.2.3 Binarization

- Converts numerical values into **binary values (0 or 1)** based on a **threshold**.
- Rule: if `x > threshold` → 1, else → 0.
- Used when only the presence/absence or "above/below a cutoff" matters, not the exact magnitude.
- Example: converting a "number of purchases" feature into "has purchased (1) / has not purchased (0)".

#### 7.2.4 Comparison Table — Feature Scaling Techniques

| Technique | Output Range | Sensitive to Outliers? | Common Use Case |
|---|---|---|---|
| Standardization | Unbounded (mean 0, std 1) | Less sensitive | Gradient-based models, PCA, SVM |
| Normalization | Fixed [0, 1] | Highly sensitive | Neural networks, KNN, image data |
| Binarization | {0, 1} only | N/A (threshold-based) | Presence/absence features |

---

## 7. Categorical Data & Encoding

### 7.1 What is Categorical Data?

- Data that represents **categories/labels** rather than numbers (e.g., "Red/Green/Blue", "Male/Female", "City names").
- Two sub-types:
  - **Nominal** — no inherent order (e.g., colors, gender).
  - **Ordinal** — has a meaningful order (e.g., Low/Medium/High).

### 7.2 Label Encoding

- Converts each category into a **unique integer** (e.g., Red=0, Green=1, Blue=2).
- Simple and memory-efficient, but introduces a **false sense of order/rank** between categories — risky for nominal data used in distance-based or linear models.
- Best suited for **ordinal data** where order is meaningful.

### 7.3 One-Hot Encoding

- Converts each category into a **separate binary column** (0/1), with only one column "hot" (1) per row.
- Example: "Color" with Red/Green/Blue → 3 new columns: `Color_Red`, `Color_Green`, `Color_Blue`.
- **Advantages:**
  - No false ordinal relationship is introduced between categories.
  - Works well with linear models, distance-based models, and neural networks.
  - Keeps categories independent of each other.
- **Trade-off:** increases dimensionality when a feature has many categories ("curse of dimensionality").

---

## 8. Simple Imputer

- `SimpleImputer` is a preprocessing tool used to **handle missing values** in a dataset.
- Replaces missing values using a chosen strategy:
  - **Mean** — replaces missing numeric values with the column mean.
  - **Median** — replaces with the column median (robust to outliers).
  - **Most frequent (mode)** — replaces with the most common value (works for categorical data too).
  - **Constant** — replaces with a fixed value you specify.
- Ensures the dataset has no missing entries before feeding it into an ML model (most algorithms can't handle NaN values directly).

---

## 9. Clustering

### 9.1 What is Clustering?

- An **unsupervised learning** technique that groups similar data points together into **clusters**, based on similarity/distance, without using labeled data.

### 9.2 Types of Clustering

1. **Partition-based Clustering** — divides data into a fixed number of non-overlapping clusters (e.g., K-Means).
2. **Hierarchical Clustering** — builds a tree-like structure of nested clusters (Agglomerative/Divisive).
3. **Density-based Clustering** — forms clusters based on dense regions of data points, can find arbitrarily shaped clusters and detect noise/outliers (e.g., DBSCAN).
4. **Distribution-based Clustering** — assumes data is generated from a mixture of statistical distributions (e.g., Gaussian Mixture Models).

---

## 10. K-Means Clustering

### 10.1 Rough Idea of How K-Means Works (Conceptual)

1. Choose the number of clusters, **k**.
2. Randomly initialize **k** centroids (cluster centers).
3. Assign each data point to its **nearest centroid** (based on distance, usually Euclidean).
4. Recalculate each centroid as the **mean** of all points assigned to it.
5. Repeat steps 3–4 until centroids stop moving significantly (convergence).
- End result: data is partitioned into k clusters, each represented by its centroid.

### 10.2 Library Used

- `scikit-learn` → `from sklearn.cluster import KMeans`

### 10.3 Problems in K-Means

- Need to **pre-specify k** (number of clusters) in advance.
- Sensitive to **initial centroid placement** — can converge to a poor/local optimum.
- Assumes clusters are **spherical and similarly sized** — struggles with irregularly shaped or unevenly sized clusters.
- Sensitive to **outliers** (they can pull centroids away from the true cluster center).
- Sensitive to **feature scale** — features should be scaled before applying K-Means.

### 10.4 Elbow Method

- A technique to choose the optimal value of **k**.
- Plots **k** (x-axis) vs **WCSS** (Within-Cluster Sum of Squares, y-axis) — WCSS measures how tightly points are clustered around their centroid.
- As k increases, WCSS decreases; the "elbow point" (where the decrease sharply slows down) is chosen as the optimal k.

### 10.5 Silhouette Coefficient

- A metric to evaluate **how well-separated and cohesive** the clusters are, ranges from **−1 to +1**.
- Formula (conceptual, not for calculation):

  **s = (b − a) / max(a, b)**

  - `a` = average distance between a point and other points in the **same cluster** (cohesion)
  - `b` = average distance between a point and points in the **nearest different cluster** (separation)
  - `s` closer to **+1** → well-clustered; closer to **0** → overlapping clusters; negative → likely misclassified point.

### 10.6 Advantages of K-Means

- Simple, fast, and easy to implement/interpret.
- Scales well to large datasets.
- Works well when clusters are roughly spherical and well-separated.

### 10.7 Disadvantages of K-Means

- Must choose k in advance.
- Sensitive to initialization and outliers.
- Struggles with non-spherical/overlapping clusters.
- Assumes clusters of similar size and density.

### 10.8 Why `random_state = 42`?

- `random_state` fixes the **seed** for K-Means' random centroid initialization, making results **reproducible** across runs.
- `42` itself has no mathematical significance — it's a commonly used, arbitrary convention in the ML community (popular culture reference from "The Hitchhiker's Guide to the Galaxy," where 42 is "the answer to life, the universe, and everything"). Any fixed integer would work identically for reproducibility.

---

## 11. Hierarchical Clustering

- Builds a hierarchy (tree) of clusters, visualized using a **dendrogram**, without needing to pre-specify the number of clusters.

### 11.1 Types of Hierarchical Clustering

1. **Agglomerative (Bottom-Up)**
   - Starts with each data point as its own individual cluster.
   - Repeatedly merges the two closest clusters until only one cluster (or desired number) remains.
2. **Divisive (Top-Down)**
   - Starts with all data points in one single cluster.
   - Repeatedly splits clusters into smaller ones until each point is its own cluster (or desired number is reached).

### 11.2 Agglomerative vs Divisive — Comparison Table

| Aspect | Agglomerative (Bottom-Up) | Divisive (Top-Down) |
|---|---|---|
| Starting point | Each point is its own cluster | All points in one cluster |
| Direction | Merges clusters upward | Splits clusters downward |
| Computational cost | Generally lower, more common | Generally higher, less common |
| Usage in practice | Widely used | Rarely used |

### 11.3 Hierarchical Clustering vs K-Means — Comparison Table

| Aspect | Hierarchical Clustering | K-Means |
|---|---|---|
| Number of clusters | Not needed upfront (decide by cutting dendrogram) | Must be specified upfront (k) |
| Output | Tree/dendrogram of nested clusters | Flat set of k clusters |
| Scalability | Slower on large datasets | Faster, scales better |
| Cluster shape | Can handle more varied/nested structures | Assumes spherical clusters |
| Reproducibility | Deterministic (no random initialization) | Depends on centroid initialization (random_state) |

### 11.4 Linkage Methods (for Agglomerative Clustering)

- Determines **how distance between two clusters** is measured when deciding which to merge:
  1. **Single Linkage** — distance between the **closest** pair of points from two clusters (can cause "chaining" of elongated clusters).
  2. **Complete Linkage** — distance between the **farthest** pair of points from two clusters (tends to form more compact, balanced clusters).
  3. **Average Linkage** — average distance between **all pairs** of points across the two clusters.
  4. **Ward Linkage** — merges the pair of clusters that leads to the **minimum increase in total within-cluster variance** (tends to create clusters of similar size).

---

## 12. Naive Bayes & Bayes' Theorem

### 12.1 Bayes' Theorem

- A formula to calculate the **conditional probability** of an event, based on prior knowledge of related conditions.
- Formula (conceptual — recognize what it represents, not for calculation):

  **P(A|B) = [P(B|A) × P(A)] / P(B)**

  - `P(A|B)` = probability of A given B has occurred (posterior)
  - `P(B|A)` = probability of B given A has occurred (likelihood)
  - `P(A)` = prior probability of A
  - `P(B)` = prior probability of B (evidence)

### 12.2 Naive Bayes Classifier

- A **supervised classification algorithm** based on Bayes' Theorem.
- Called "naive" because it assumes all features are **independent of each other** given the class — an assumption rarely true in real life, but the algorithm still performs well in practice.
- Commonly used for text classification (spam detection, sentiment analysis).

### 12.3 Types of Naive Bayes

- **Gaussian Naive Bayes** — assumes continuous features follow a **normal (Gaussian) distribution**; most common type for numerical/continuous data.
- **Multinomial Naive Bayes** — used for discrete/count data (e.g., word counts in text classification).
- **Bernoulli Naive Bayes** — used for binary/boolean features (e.g., word present/absent).

---

## 13. Decision Tree

- A **supervised learning algorithm** used for both classification and regression, structured as a tree of decisions based on feature values.
- Key question: **how does the algorithm decide which feature to split on at each node?** → measured using **Entropy** and **Information Gain** (or **Gini Index**).

### 13.1 Entropy

- Measures the **impurity/randomness/disorder** in a dataset — how mixed the classes are.
- Formula (conceptual):

  **H(S) = − Σ pᵢ log₂(pᵢ)**

  - `S` = dataset/subset
  - `pᵢ` = proportion of data points belonging to class `i`
  - Entropy = 0 → pure node (all same class); Entropy = 1 (max, for 2 classes) → maximum impurity (evenly mixed).

### 13.2 Information Gain

- Measures the **reduction in entropy** after splitting a dataset on a particular feature — used to pick the best feature to split on.
- Formula (conceptual):

  **IG(S, A) = Entropy(S) − Σ (|Sᵥ| / |S|) × Entropy(Sᵥ)**

  - `S` = original dataset before split
  - `A` = feature being considered for the split
  - `Sᵥ` = subset of S where feature A has value v
  - Higher Information Gain → better feature to split on (the algorithm picks the feature with the **highest** Information Gain at each node).

### 13.3 ID3 Algorithm

- One of the earliest decision tree algorithms; builds the tree using **Entropy and Information Gain**.
- How it works (rough idea):
  1. Calculate entropy of the entire dataset.
  2. Calculate information gain for every feature.
  3. Choose the feature with the **highest information gain** as the splitting node (root, then recursively for child nodes).
  4. Repeat for each branch until all data at a node belongs to one class, or no features remain, or a stopping condition is met.
  5. Result: a tree where each internal node is a feature test and each leaf is a class label.

### 13.4 Gini Index

- An alternative impurity measure to Entropy, used by algorithms like CART (used internally in scikit-learn's Decision Tree).
- Formula (conceptual):

  **Gini(S) = 1 − Σ pᵢ²**

  - `pᵢ` = proportion of data points belonging to class `i`
  - Gini = 0 → pure node; higher Gini → more impurity. Feature with **lowest** Gini after split is chosen.

### 13.5 Entropy vs Gini Index — Comparison Table

| Aspect | Entropy | Gini Index |
|---|---|---|
| Formula basis | Logarithmic (−Σ pᵢ log₂ pᵢ) | Squared proportions (1 − Σ pᵢ²) |
| Computation | Slightly slower (log calculation) | Faster (no log) |
| Range | 0 to 1 (for binary classes) | 0 to 0.5 (for binary classes) |
| Used in | ID3, C4.5 | CART (scikit-learn default) |
| Sensitivity | Slightly more sensitive to class imbalance | Slightly less sensitive |

---

## 14. Random Forest

### 14.1 Why Not Just Use a Normal Decision Tree?

- A single Decision Tree tends to **overfit** the training data — it can become overly complex and memorize noise, leading to poor generalization on new data.
- Random Forest solves this by combining **many decision trees** to make more robust, generalized predictions.

### 14.2 Decision Tree vs Random Forest — Comparison Table

| Aspect | Decision Tree | Random Forest |
|---|---|---|
| Structure | Single tree | Ensemble (many trees) |
| Overfitting risk | High | Low (averaging reduces variance) |
| Accuracy | Lower, more variable | Generally higher, more stable |
| Interpretability | Easy to visualize/interpret | Harder to interpret (black-box-ish) |
| Speed | Faster to train | Slower (multiple trees) |

### 14.3 Rough Idea of How Random Forest Works

1. Create multiple random subsets of the training data (with replacement — **bagging**).
2. Train a separate Decision Tree on each subset, also considering only a random subset of features at each split.
3. For classification: each tree "votes" for a class, and the **majority vote** becomes the final prediction.
4. For regression: predictions from all trees are **averaged**.

### 14.4 When to Use Random Forest

- When a single Decision Tree overfits the data.
- When you need higher accuracy and robustness without extensive tuning.
- When feature importance ranking is needed (Random Forest can estimate this).
- Works well on both classification and regression tasks with tabular data.

### 14.5 Important Terms

- **`n_estimators`** — the number of decision trees to build in the forest; more trees generally improve stability (up to a point) but increase computation time.
- **Bagging (Bootstrap Aggregating)** — the technique of training each tree on a random sample (with replacement) of the data.
- **Feature Importance** — a score indicating how much each feature contributed to reducing impurity across all trees.

---

## 15. Model Evaluation Metrics (Classification)

### 15.1 Confusion Matrix

- A table summarizing prediction results, comparing **actual vs predicted** classes:

| | Predicted Positive | Predicted Negative |
|---|---|---|
| **Actual Positive** | True Positive (TP) | False Negative (FN) |
| **Actual Negative** | False Positive (FP) | True Negative (TN) |

### 15.2 Accuracy

- Proportion of total predictions that were correct.
- Formula (conceptual): **Accuracy = (TP + TN) / (TP + TN + FP + FN)**
- Can be misleading on imbalanced datasets.

### 15.3 Precision

- Of all predicted positives, how many were actually positive.
- Formula (conceptual): **Precision = TP / (TP + FP)**
- Important when the cost of a **false positive** is high (e.g., spam detection).

### 15.4 Recall (Sensitivity)

- Of all actual positives, how many were correctly predicted.
- Formula (conceptual): **Recall = TP / (TP + FN)**
- Important when the cost of a **false negative** is high (e.g., disease detection).

### 15.5 F1 Score

- The **harmonic mean** of Precision and Recall — balances both when there's a trade-off between them.
- Formula (conceptual): **F1 = 2 × (Precision × Recall) / (Precision + Recall)**
- Useful when classes are imbalanced and both false positives and false negatives matter.

---

## 16. Formal Learning Model & PAC Learning

### 16.1 Formal Learning Model

- A theoretical framework describing learning as: given a **hypothesis class H**, and training data drawn i.i.d. (independently, identically distributed) from an unknown distribution **D**, the algorithm outputs a hypothesis **h ∈ H** that minimizes error on unseen data from the same distribution.

### 16.2 PAC Learning (Probably Approximately Correct)

- A formal framework defining when a concept/hypothesis class is "learnable."
- A class is **PAC learnable** if there's an algorithm that, given enough training samples, outputs a hypothesis that is **"approximately correct"** (error ≤ ε) with **"high probability"** (≥ 1 − δ), for any data distribution.
- Core idea: with enough data, the learned hypothesis will very likely be close to the true concept.

---

## 17. Bias-Complexity Trade-off

- **Bias** — error from overly simplistic assumptions in the model (leads to underfitting).
- **Variance** — error from the model being overly sensitive to small fluctuations in training data (leads to overfitting).
- **Trade-off**: increasing model complexity → lowers bias but raises variance, and vice versa. The goal is to find the sweet spot minimizing total error.
- Conceptually: **Total Error ≈ Bias² + Variance + Irreducible Error**.

---

## 18. VC Dimension

- **VC (Vapnik–Chervonenkis) Dimension** measures the **capacity/complexity** of a hypothesis class.
- Defined as the largest number of points that the hypothesis class can **"shatter"** — i.e., correctly classify in every possible labeling/combination.
- Higher VC Dimension → more complex hypothesis class → more flexible but higher risk of overfitting.

---

## 19. Non-Uniform Learnability

- A relaxation of PAC learning where the number of samples needed can depend on the **specific hypothesis** being considered, not just the hypothesis class as a whole.
- Allows assigning different "preference"/complexity weights to different hypotheses (e.g., simpler hypotheses need fewer samples to trust).

---

## 20. Stability

- **Algorithmic stability** measures how much a learning algorithm's output changes when a single training example is added, removed, or altered.
- A more **stable** algorithm tends to **generalize better** — small data changes shouldn't drastically change the learned model.
- Closely related to regularization — regularized models are typically more stable.

---

## 21. Overfitting & Underfitting

- **Overfitting** — model performs very well on training data but poorly on unseen/test data; it has "memorized" noise/details instead of learning the general pattern (high variance).
- **Underfitting** — model performs poorly on both training and test data; it's too simple to capture the underlying pattern (high bias).

---

## 22. Regularization

- A technique that adds a **penalty term** to the loss function to discourage overly complex models and reduce overfitting.
- **L1 Regularization (Lasso)** — adds sum of **absolute** weights as penalty; can shrink some weights to exactly zero (acts as feature selection).
- **L2 Regularization (Ridge)** — adds sum of **squared** weights as penalty; shrinks all weights but rarely to exactly zero.
- **Elastic Net** — combines both L1 and L2 penalties.

---

## 23. Model Selection & Validation

### 23.1 Model Selection

- The process of choosing the best model and/or best hyperparameters among multiple candidates, based on performance on validation data.

### 23.2 Validation

- Evaluating a trained model's performance on a **held-out dataset** (not used during training) to estimate how it will perform on unseen data.

### 23.3 Cross-Validation

- A resampling technique to more reliably assess how well a model generalizes, by repeatedly training/testing on different splits of the data rather than a single train/test split.

### 23.4 K-Fold Cross-Validation

- Data is split into **k equal folds**.
- The model is trained on **k−1 folds** and tested on the **remaining 1 fold** — repeated **k times**, each time with a different fold as the test set.
- Final performance = average of all k iterations' results — gives a more robust estimate than a single train/test split.

---

## 24. Scikit-learn Library Overview

- **Scikit-learn** is a popular open-source Python library for Machine Learning, providing tools for preprocessing, model building, evaluation, and model selection.
- Key modules/areas (conceptual overview, not code-focused):
  - `sklearn.preprocessing` — scaling, encoding (StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder, Binarizer).
  - `sklearn.impute` — handling missing values (SimpleImputer).
  - `sklearn.model_selection` — splitting/validating data (train_test_split, cross_val_score, KFold, GridSearchCV).
  - `sklearn.linear_model` — regression models (LinearRegression, Ridge, Lasso).
  - `sklearn.tree` — Decision Trees.
  - `sklearn.ensemble` — Random Forest, AdaBoost, Gradient Boosting.
  - `sklearn.svm` — Support Vector Machines.
  - `sklearn.neighbors` — KNN.
  - `sklearn.naive_bayes` — Naive Bayes variants.
  - `sklearn.cluster` — K-Means, Hierarchical Clustering.
  - `sklearn.decomposition` — PCA, Kernel PCA.
  - `sklearn.discriminant_analysis` — LDA.
  - `sklearn.metrics` — evaluation metrics (accuracy, precision, recall, F1, MAE, MSE, R², silhouette score).

---

## 25. Regression

### 25.1 Linear Regression

- Models the relationship between a dependent variable **y** and independent variable(s) **x** as a **straight line**: y = β₀ + β₁x (simple) or y = β₀ + β₁x₁ + β₂x₂ + ... (multiple).
- Used when the relationship between variables is approximately linear.

### 25.2 Other Types of Regression

- **Simple Linear Regression** — one independent variable.
- **Multiple Linear Regression** — two or more independent variables.
- **Polynomial Regression** — models non-linear relationships by adding polynomial terms (x², x³, etc.) of the features.
- **Ridge Regression** — Linear Regression with L2 regularization.
- **Lasso Regression** — Linear Regression with L1 regularization.
- **Elastic Net Regression** — combines L1 + L2 regularization.

### 25.3 Non-Linear Regression

- Models relationships that **cannot** be represented as a straight line (e.g., exponential, logarithmic, or other curved patterns) by fitting a non-linear function to the data.

---

## 26. K-Nearest Neighbors (KNN)

- A **supervised**, **instance-based ("lazy")** learning algorithm — it doesn't build an explicit model during training; instead, it stores the training data and makes predictions at query time.
- Classifies a new point based on the **majority class** among its **k nearest neighbors** (classification), or the **average value** of its k nearest neighbors (regression), based on distance (commonly Euclidean).

### 26.1 Advantages of KNN

- Simple to understand and implement.
- No explicit training phase (lazy learner).
- Naturally handles multi-class classification.
- Makes no assumptions about the underlying data distribution.

### 26.2 Disadvantages of KNN

- Computationally expensive at prediction time (must compute distance to all training points).
- Sensitive to irrelevant features and feature scale (requires feature scaling).
- Performance depends heavily on the choice of **k**.
- Struggles with high-dimensional data ("curse of dimensionality").

---

## 27. Support Vector Machine (SVM)

- A **supervised** algorithm that finds the **optimal hyperplane** that best separates data points of different classes.

- **Hyperplane** — the decision boundary that separates classes in feature space.
- **Margin** — the distance between the hyperplane and the nearest data points of each class; SVM aims to **maximize** this margin (also called a "maximum margin classifier").
- **Support Vectors** — the data points closest to the hyperplane; they "support"/define the position of the margin and hyperplane.
- **Kernel** — a function used to transform data into a higher-dimensional space, making it linearly separable when it isn't in the original space (the "kernel trick"). Common kernels: Linear, Polynomial, RBF (Gaussian).

---

## 28. Ensemble Learning

- Combines predictions from **multiple models ("weak learners")** to produce a stronger, more accurate overall model.

### 28.1 Types of Ensemble Learning

1. **Bagging (Bootstrap Aggregating)** — trains multiple models **independently** on random subsets (with replacement) of the data, then combines results via voting (classification) or averaging (regression). Example: Random Forest.
2. **Boosting** — trains models **sequentially**, where each new model focuses on correcting the errors of the previous ones; final prediction is a weighted combination. Examples: AdaBoost, Gradient Boosting.

### 28.2 Gradient Boosting

- A boosting technique where each new weak learner is trained to predict the **residual errors** (based on the gradient of the loss function) of the combined previous learners.
- Models are added sequentially, each correcting the shortcomings of the ensemble so far.

### 28.3 AdaBoost (Adaptive Boosting) — Detailed

- A boosting algorithm that adaptively adjusts the weights of training samples, focusing more on samples that were previously misclassified.

**How AdaBoost Works (step-by-step, conceptual):**
1. Assign **equal weights** to all training samples initially.
2. Train a **weak learner** (commonly a **decision stump** — see below) on the weighted data.
3. Calculate the weak learner's **error rate**, and compute its **"say"/weight** in the final vote — more accurate learners get a higher say.
4. **Increase the weights** of misclassified samples (so the next learner pays more attention to them) and **decrease the weights** of correctly classified samples.
5. Repeat steps 2–4 for a set number of iterations, training a new weak learner each time on the re-weighted data.
6. Final prediction = **weighted majority vote** (classification) or weighted sum (regression) of all weak learners.

**What is a Decision Stump?**
- A **decision stump** is a Decision Tree with just **one split (depth = 1)** — it makes a decision based on only a single feature.
- It's a very simple, "weak" learner (only slightly better than random guessing) — commonly used as the base learner in AdaBoost, since boosting combines many weak learners into a strong one.

---

## 29. Regression Metrics

- **Mean Absolute Error (MAE)** — average of the absolute differences between predicted and actual values.
  - Formula (conceptual): **MAE = (1/n) Σ |yᵢ − ŷᵢ|**
- **Mean Squared Error (MSE)** — average of the squared differences; penalizes larger errors more heavily than MAE.
  - Formula (conceptual): **MSE = (1/n) Σ (yᵢ − ŷᵢ)²**
- **Root Mean Squared Error (RMSE)** — square root of MSE; brings the error metric back to the same units as the original target variable.
  - Formula (conceptual): **RMSE = √MSE**
- **R² Score (Coefficient of Determination)** — proportion of variance in the target variable explained by the model; ranges up to 1 (1 = perfect fit; can be negative for a poor model).
  - Formula (conceptual): **R² = 1 − (SS_residual / SS_total)**

---

## 30. Dimensionality Reduction

### 30.1 PCA (Principal Component Analysis)

- An **unsupervised** technique that transforms correlated features into a smaller set of **uncorrelated "principal components"**, which capture the maximum variance in the data.

### 30.2 Kernel PCA

- An extension of PCA using the **kernel trick**, allowing it to handle **non-linearly separable** data by implicitly projecting it into a higher-dimensional space before reducing dimensions.

### 30.3 LDA (Linear Discriminant Analysis)

- A **supervised** dimensionality reduction technique — unlike PCA (which only maximizes variance), LDA finds a projection that **maximizes separability between known classes**.

### 30.4 Random Projection

- A dimensionality reduction technique that projects data into a lower-dimensional space using a **random matrix**, approximately preserving distances between points (based on the Johnson–Lindenstrauss lemma).
- Much **faster** than PCA, especially useful for very high-dimensional data.

---

## 31. Information Theory

### 31.1 Entropy (Information Theory Context)

- Measures the **uncertainty or information content** of a random variable/distribution — the broader Shannon information-theory concept underlying the entropy used in Decision Trees.

### 31.2 Cross Entropy

- Measures the **difference between two probability distributions** — typically the predicted probability distribution vs. the true/actual distribution.
- Widely used as a **loss function** for classification models (also called "log loss").
- Formula (conceptual): **Cross-Entropy = − Σ yᵢ log(ŷᵢ)**
  - `yᵢ` = true label/distribution, `ŷᵢ` = predicted probability.

---

## 32. Time Series Forecasting

### 32.1 Core Components

- **Trend** — the long-term upward or downward movement in the data over time.
- **Seasonality** — repeating patterns/cycles that occur at regular intervals (daily, weekly, yearly, etc.).
- **Noise** — random, irregular fluctuations in the data with no discernible pattern.

### 32.2 ARIMA (AutoRegressive Integrated Moving Average)

- A statistical model for time series forecasting, combining three components:
  - **AR (AutoRegressive)** — predicts future values using a linear combination of past values.
  - **I (Integrated)** — applies differencing to the data to remove trend and make it stationary.
  - **MA (Moving Average)** — uses past forecast errors to improve current predictions.
- Represented as **ARIMA(p, d, q)**, where `p` = AR order, `d` = differencing order, `q` = MA order.

---

## 33. Collinearity & Multicollinearity

- **Collinearity** — when **two** independent features/variables are highly correlated with each other.
- **Multicollinearity** — when **three or more** independent variables are highly correlated, making it difficult to isolate each feature's individual effect on the target and potentially destabilizing regression coefficients.

---

## Algorithms Studied So Far
*(This table will be updated every time a new algorithm topic is covered — lists the algorithm and which type of Machine Learning it belongs to.)*

| Algorithm | Type of Machine Learning |
|---|---|
| K-Means Clustering | Unsupervised Learning (Partition-based Clustering) |
| Hierarchical Clustering (Agglomerative/Divisive) | Unsupervised Learning (Hierarchical Clustering) |
| Naive Bayes (Gaussian/Multinomial/Bernoulli) | Supervised Learning (Classification) |
| Decision Tree | Supervised Learning (Classification/Regression) |
| Random Forest | Supervised Learning (Ensemble Classification/Regression) |
| Linear Regression (Simple/Multiple/Polynomial/Ridge/Lasso/Elastic Net) | Supervised Learning (Regression) |
| K-Nearest Neighbors (KNN) | Supervised Learning (Classification/Regression) |
| Support Vector Machine (SVM) | Supervised Learning (Classification/Regression) |
| AdaBoost | Supervised Learning (Ensemble — Boosting) |
| Gradient Boosting | Supervised Learning (Ensemble — Boosting) |
| PCA / Kernel PCA | Unsupervised Learning (Dimensionality Reduction) |
| LDA (Linear Discriminant Analysis) | Supervised Learning (Dimensionality Reduction) |
| Random Projection | Unsupervised Learning (Dimensionality Reduction) |
| ARIMA | Supervised/Statistical (Time Series Forecasting) |

---

## Library & Import Cheat Sheet
*(to be filled in as algorithms are covered)*

| Algorithm/Technique | Library | Import Statement |
|---|---|---|
| Standardization (Z-score) | scikit-learn | `from sklearn.preprocessing import StandardScaler` |
| Normalization (Min-Max) | scikit-learn | `from sklearn.preprocessing import MinMaxScaler` |
| Binarization | scikit-learn | `from sklearn.preprocessing import Binarizer` |
| Label Encoding | scikit-learn | `from sklearn.preprocessing import LabelEncoder` |
| One-Hot Encoding | scikit-learn | `from sklearn.preprocessing import OneHotEncoder` |
| Simple Imputer (missing values) | scikit-learn | `from sklearn.impute import SimpleImputer` |
| K-Means Clustering | scikit-learn | `from sklearn.cluster import KMeans` |
| Hierarchical (Agglomerative) Clustering | scikit-learn | `from sklearn.cluster import AgglomerativeClustering` |
| Silhouette Coefficient (evaluation metric) | scikit-learn | `from sklearn.metrics import silhouette_score` |
| Gaussian Naive Bayes | scikit-learn | `from sklearn.naive_bayes import GaussianNB` |
| Multinomial Naive Bayes | scikit-learn | `from sklearn.naive_bayes import MultinomialNB` |
| Bernoulli Naive Bayes | scikit-learn | `from sklearn.naive_bayes import BernoulliNB` |
| Decision Tree (Classifier/Regressor) | scikit-learn | `from sklearn.tree import DecisionTreeClassifier` / `DecisionTreeRegressor` |
| Random Forest (Classifier/Regressor) | scikit-learn | `from sklearn.ensemble import RandomForestClassifier` / `RandomForestRegressor` |
| Confusion Matrix | scikit-learn | `from sklearn.metrics import confusion_matrix` |
| Accuracy, Precision, Recall, F1 Score | scikit-learn | `from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score` |
| Train/Test Split | scikit-learn | `from sklearn.model_selection import train_test_split` |
| Cross-Validation / K-Fold | scikit-learn | `from sklearn.model_selection import cross_val_score, KFold` |
| Grid Search (hyperparameter tuning) | scikit-learn | `from sklearn.model_selection import GridSearchCV` |
| Linear Regression | scikit-learn | `from sklearn.linear_model import LinearRegression` |
| Ridge Regression | scikit-learn | `from sklearn.linear_model import Ridge` |
| Lasso Regression | scikit-learn | `from sklearn.linear_model import Lasso` |
| Elastic Net Regression | scikit-learn | `from sklearn.linear_model import ElasticNet` |
| Polynomial Regression (feature transform) | scikit-learn | `from sklearn.preprocessing import PolynomialFeatures` |
| K-Nearest Neighbors | scikit-learn | `from sklearn.neighbors import KNeighborsClassifier` / `KNeighborsRegressor` |
| Support Vector Machine | scikit-learn | `from sklearn.svm import SVC` / `SVR` |
| AdaBoost | scikit-learn | `from sklearn.ensemble import AdaBoostClassifier` |
| Gradient Boosting | scikit-learn | `from sklearn.ensemble import GradientBoostingClassifier` |
| Mean Absolute Error / MSE / R² Score | scikit-learn | `from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score` |
| PCA | scikit-learn | `from sklearn.decomposition import PCA` |
| Kernel PCA | scikit-learn | `from sklearn.decomposition import KernelPCA` |
| LDA (Linear Discriminant Analysis) | scikit-learn | `from sklearn.discriminant_analysis import LinearDiscriminantAnalysis` |
| Random Projection | scikit-learn | `from sklearn.random_projection import GaussianRandomProjection` |
| ARIMA | statsmodels | `from statsmodels.tsa.arima.model import ARIMA` |
