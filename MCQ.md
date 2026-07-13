# MCQ Bank — Deep Learning, CNN, Transformers & Generative AI

*(Extracted and reformatted from Sample_MCQs_DNN.docx)*

---

## Contents

- Deep Neural Networks, CNN & RNN — Fundamentals (54 questions)
- Advanced Deep Learning Concepts (19 questions)
- Generative AI (10 questions)
- Transformer Models (10 questions)
- Convolutional Neural Networks (CNNs) (10 questions)
- Convolutional Layers (10 questions)
- Pooling Layers (10 questions)

---

## Deep Neural Networks, CNN & RNN — Fundamentals

**Q1. Which of the following is a characteristic of Deep Neural Networks (DNNs)?**

- A) Shallow layers
- B) High computational cost
- C) Low accuracy
- D) Lack of activation functions

Answer: **B) High computational cost**

---

**Q2. What is the primary purpose of activation functions in neural networks?**

- A) To initialize weights
- B) To introduce non-linearity
- C) To update biases
- D) To reduce overfitting

Answer: **B) To introduce non-linearity**

---

**Q3. In CNNs, what operation is used to reduce the spatial dimensions of feature maps?**

- A) Convolution
- B) Pooling
- C) ReLU
- D) Softmax

Answer: **B) Pooling**

---

**Q4. Which of the following activation functions is commonly used in DNNs for its simplicity and effectiveness?**

- A) Sigmoid
- B) Tanh
- C) ReLU
- D) Softmax

Answer: **C) ReLU**

---

**Q5. What is the role of the Softmax function in a neural network?**

- A) To perform convolutions
- B) To reduce the dimensionality of data
- C) To classify multi-class outputs
- D) To introduce non-linearity

Answer: **C) To classify multi-class outputs**

---

**Q6. Which layer in a CNN is responsible for detecting features such as edges and textures?**

- A) Input layer
- B) Convolutional layer
- C) Pooling layer
- D) Fully connected layer

Answer: **B) Convolutional layer**

---

**Q7. What is overfitting in the context of neural networks?**

- A) A model that performs well on training data but poorly on new data
- B) A model that generalizes well to new data
- C) A model with too few parameters
- D) A model with a high learning rate

Answer: **A) A model that performs well on training data but poorly on new data**

---

**Q8. Which technique can help reduce overfitting in DNNs?**

- A) Increasing the learning rate
- B) Adding more layers
- C) Using dropout
- D) Removing regularization

Answer: **C) Using dropout**

---

**Q9. In a neural network, what does the term "epoch" refer to?**

- A) One forward pass
- B) One backward pass
- C) One complete pass through the entire training dataset
- D) One parameter update

Answer: **C) One complete pass through the entire training dataset**

---

**Q10. Which type of neural network is specifically designed for sequential data such as time series?**

- A) CNN
- B) DNN
- C) RNN
- D) GAN

Answer: **C) RNN**

---

**Q11. What is the primary difference between CNNs and DNNs?**

- A) CNNs use fully connected layers only
- B) CNNs are used for image data, while DNNs are used for structured
- C) CNNs use convolutional and pooling layers, while DNNs use fully
- D) CNNs have fewer parameters than DNNs

Answer: **C) CNNs use convolutional and pooling layers, while DNNs use fully**

---

**Q12. In CNNs, what does the term "stride" refer to?**

- A) The size of the filters
- B) The number of layers in the network
- C) The step size for moving the filter across the input
- D) The depth of the convolutional layer

Answer: **C) The step size for moving the filter across the input**

---

**Q13. Which of the following is a common loss function used for binary classification tasks?**

- A) Mean Squared Error
- B) Cross-Entropy Loss
- C) Hinge Loss
- D) Categorical Cross-Entropy

Answer: **B) Cross-Entropy Loss**

---

**Q14. In backpropagation, what is the purpose of the gradient descent algorithm?**

- A) To initialize the weights
- B) To update the biases only
- C) To minimize the loss function
- D) To normalize the input data

Answer: **C) To minimize the loss function**

---

**Q15. What is the vanishing gradient problem?**

- A) Gradients that become too large and cause instability
- B) Gradients that disappear, leading to very small updates
- C) Gradients that increase the learning rate
- D) Gradients that cause overfitting

Answer: **B) Gradients that disappear, leading to very small updates**

---

**Q16. In CNNs, what is the function of padding?**

- A) To increase the number of parameters
- B) To reduce the number of layers
- C) To control the spatial dimensions of the output
- D) To increase the learning rate

Answer: **C) To control the spatial dimensions of the output**

---

**Q17. Which optimizer is known for adapting the learning rate for each parameter during training?**

- A) Stochastic Gradient Descent
- B) RMSprop
- C) Adam
- D) Adagrad

Answer: **C) Adam**

---

**Q18. What is the purpose of using a validation set in neural network training?**

- A) To adjust the learning rate
- B) To fine-tune hyperparameters and prevent overfitting
- C) To reduce the number of training epochs
- D) To increase the size of the training set

Answer: **B) To fine-tune hyperparameters and prevent overfitting**

---

**Q19. Which of the following describes batch normalization?**

- A) A technique to normalize the output layer
- B) A method to reduce the variance of the input data
- C) A technique to normalize the inputs of each layer to improve
- D) A method to initialize weights

Answer: **C) A technique to normalize the inputs of each layer to improve**

---

**Q20. What is the main advantage of using Recurrent Neural Networks (RNNs) over traditional DNNs?**

- A) RNNs can handle non-sequential data better
- B) RNNs can process variable-length input sequences
- C) RNNs are easier to train
- D) RNNs have fewer parameters

Answer: **B) RNNs can process variable-length input sequences**

---

**Q21. Which pooling operation retains the maximum value from each patch of the feature map?**

- A) Average pooling
- B) Max pooling
- C) Sum pooling
- D) Min pooling

Answer: **B) Max pooling**

---

**Q22. What is the primary challenge addressed by Long Short-Term Memory (LSTM) networks?**

- A) Overfitting
- B) Vanishing gradients in RNNs
- C) High computational cost
- D) Lack of activation functions

Answer: **B) Vanishing gradients in RNNs**

---

**Q23. Which of the following techniques can be used to prevent overfitting in neural networks?**

- A) Using a high learning rate
- B) Data augmentation
- C) Reducing the number of layers
- D) Removing dropout layers

Answer: **B) Data augmentation**

---

**Q24. What does the term "kernel" refer to in the context of CNNs?**

- A) The input data
- B) The activation function
- C) The filter used in the convolution operation
- D) The fully connected layer

Answer: **C) The filter used in the convolution operation**

---

**Q25. Which layer type in a neural network is responsible for combining features learned by previous layers to make final predictions?**

- A) Convolutional layer
- B) Pooling layer
- C) Dropout layer
- D) Fully connected layer

Answer: **D) Fully connected layer**

---

**Q26. In a neural network, what is a weight matrix?**

- A) The initial input data
- B) A matrix that stores activation values
- C) A matrix that connects neurons and holds the learned parameters
- D) A matrix that normalizes inputs

Answer: **C) A matrix that connects neurons and holds the learned parameters**

---

**Q27. What is the primary function of the loss function in neural networks?**

- A) To initialize the network
- B) To measure the error of the model's predictions
- C) To update the learning rate
- D) To combine the input data

Answer: **B) To measure the error of the model's predictions**

---

**Q28. What is the purpose of using dropout in a neural network?**

- A) To increase the size of the training data
- B) To reduce computational cost
- C) To prevent overfitting by randomly dropping units during training
- D) To initialize weights

Answer: **C) To prevent overfitting by randomly dropping units during training**

---

**Q29. In the context of CNNs, what is a feature map?**

- A) The original input image
- B) The output of the pooling layer
- C) The result of applying a filter to the input data
- D) The final prediction of the network

Answer: **C) The result of applying a filter to the input data**

---

**Q30. Which neural network architecture is best suited for image classification tasks?**

- A) RNN
- B) CNN
- C) DNN
- D) GAN

Answer: **B) CNN**

---

**Q31. What does the term "epoch" refer to in neural network training?**

- A) One forward pass
- B) One backward pass
- C) One complete pass through the training data
- D) One update of the weights

Answer: **C) One complete pass through the training data**

---

**Q32. Which of the following is a commonly used optimization algorithm in training neural networks?**

- A) k-means
- B) Apriori
- C) Adam
- D) Dijkstra's algorithm

Answer: **C) Adam**

---

**Q33. What is the function of the ReLU activation in neural networks?**

- A) To scale the input values
- B) To introduce non-linearity by setting negative values to zero
- C) To normalize the input values
- D) To reduce the dimensionality

Answer: **B) To introduce non-linearity by setting negative values to zero**

---

**Q34. Which type of layer is responsible for down-sampling the input in a CNN?**

- A) Convolutional layer
- B) Pooling layer
- C) Fully connected layer
- D) Recurrent layer

Answer: **B) Pooling layer**

---

**Q35. What is the main advantage of using batch normalization?**

- A) Faster training
- B) Increased overfitting
- C) More complex models
- D) Smaller datasets

Answer: **A) Faster training**

---

**Q36. Which component of neural networks helps to introduce non-linearity into the model?**

- A) Weight initialization
- B) Activation functions
- C) Learning rate
- D) Loss function

Answer: **B) Activation functions**

---

**Q37. What is a common technique to prevent neural networks from overfitting?**

- A) Using a low learning rate
- B) Reducing the number of layers
- C) Early stopping
- D) Removing dropout layers

Answer: **C) Early stopping**

---

**Q38. What is the function of a fully connected layer in a neural network?**

- A) To perform convolutions
- B) To combine features into final predictions
- C) To down-sample feature maps
- D) To normalize inputs

Answer: **B) To combine features into final predictions**

---

**Q39. Which of the following is NOT a type of neural network architecture?**

- A) Convolutional Neural Network (CNN)
- B) Recurrent Neural Network (RNN)
- C) Generative Adversarial Network (GAN)
- D) Support Vector Machine (SVM)

Answer: **D) Support Vector Machine (SVM)**

---

**Q40. In the context of CNNs, what is a filter or kernel?**

- A) A method for normalizing input data
- B) A set of weights used in convolution operations
- C) A pooling operation
- D) An activation function

Answer: **B) A set of weights used in convolution operations**

---

**Q41. What does "backpropagation" refer to in neural networks?**

- A) Initializing weights
- B) Forward propagation of inputs
- C) Updating weights based on the gradient of the loss
- D) Reducing the dimensionality of the data

Answer: **C) Updating weights based on the gradient of the loss**

---

**Q42. Which technique is used to address the vanishing gradient problem in deep neural networks?**

- A) Increasing the learning rate
- B) Using ReLU activation functions
- C) Adding more layers
- D) Removing regularization

Answer: **B) Using ReLU activation functions**

---

**Q43. What is a common use case for Convolutional Neural Networks (CNNs)?**

- A) Time series prediction
- B) Natural language processing
- C) Image recognition and classification
- D) Recommender systems

Answer: **C) Image recognition and classification**

---

**Q44. In neural networks, what does the term "dropout" refer to?**

- A) Dropping the learning rate
- B) Randomly dropping units during training to prevent overfitting
- C) Dropping entire layers
- D) Dropping the dataset size

Answer: **B) Randomly dropping units during training to prevent overfitting**

---

**Q45. Which activation function is defined as \\(f(x) = max(0, x)\)?**

- A) Sigmoid
- B) Tanh
- C) ReLU
- D) Softmax

Answer: **C) ReLU**

---

**Q46. What is the primary goal of using a loss function in training neural networks?**

- A) To initialize weights
- B) To measure the error of the model's predictions
- C) To update the learning rate
- D) To normalize input data

Answer: **B) To measure the error of the model's predictions**

---

**Q47. Which of the following is a common optimization algorithm used in neural network training?**

- A) K-means clustering
- B) Apriori
- C) Adam
- D) Quick sort

Answer: **C) Adam**

---

**Q48. What is the main purpose of pooling layers in CNNs?**

- A) To perform convolutions
- B) To reduce the spatial dimensions of feature maps
- C) To classify outputs
- D) To normalize input data

Answer: **B) To reduce the spatial dimensions of feature maps**

---

**Q49. In the context of CNNs, what does "stride" refer to?**

- A) The number of layers
- B) The number of neurons
- C) The step size for moving the filter across the input
- D) The activation function

Answer: **C) The step size for moving the filter across the input**

---

**Q50. What is the purpose of the ReLU activation function?**

- A) To scale the input values
- B) To introduce non-linearity by setting negative values to zero
- C) To normalize input data
- D) To reduce dimensionality

Answer: **B) To introduce non-linearity by setting negative values to zero**

---

**Q51. What is the main benefit of using deep learning over traditional machine learning algorithms?**

- A) Requires less data
- B) Automatically extracts features from raw data
- C) Faster to train
- D) Simpler models

Answer: **B) Automatically extracts features from raw data**

---

**Q52. Which neural network layer type is responsible for learning hierarchical representations of data?**

- A) Input layer
- B) Convolutional layer
- C) Pooling layer
- D) Output layer

Answer: **B) Convolutional layer**

---

**Q53. In backpropagation, what is the purpose of computing the gradient of the loss function?**

- A) To initialize weights
- B) To normalize the input data
- C) To update the model parameters
- D) To perform forward propagation

Answer: **C) To update the model parameters**

---

**Q54. Which of the following is an advantage of using the Adam optimizer?**

- A) Requires less memory
- B) Adapts the learning rate for each parameter
- C) Always converges faster
- D) Increases overfitting

Answer: **B) Adapts the learning rate for each parameter**

---

## Advanced Deep Learning Concepts

**Q55. Which type of regularization technique is known to add a penalty equal to the absolute value of the magnitude of coefficients?**

- A) L2 regularization
- B) Dropout
- C) L1 regularization
- D) Batch normalization

Answer: **C) L1 regularization**

---

**Q56. What is the main advantage of using depthwise separable convolutions in CNNs?**

- A) Increased non-linearity
- B) Reduced computational complexity and number of parameters
- C) Better generalization to new data
- D) Improved overfitting prevention

Answer: **B) Reduced computational complexity and number of parameters**

---

**Q57. Which technique is used to stabilize and accelerate the training of deep neural networks by normalizing the input of each layer?**

- A) Gradient clipping
- B) Batch normalization
- C) L2 regularization
- D) Dropout

Answer: **B) Batch normalization**

---

**Q58. In the context of training GANs, what is the primary purpose of the generator network?**

- A) To classify real images
- B) To generate data that is indistinguishable from real data
- C) To update the weights of the discriminator
- D) To reduce the dimensionality of the input data

Answer: **B) To generate data that is indistinguishable from real data**

---

**Q59. What is the primary challenge addressed by the use of residual connections in deep networks?**

- A) Vanishing gradients
- B) Overfitting
- C) Underfitting
- D) Gradient explosion

Answer: **A) Vanishing gradients**

---

**Q60. Which of the following best describes the purpose of an autoencoder?**

- A) To classify images into categories
- B) To generate new data from random noise
- C) To learn a compressed representation of data
- D) To detect objects in images

Answer: **C) To learn a compressed representation of data**

---

**Q61. In a neural network, what does the term "weight decay" refer to?**

- A) The gradual decrease of learning rate
- B) A regularization technique to penalize large weights
- C) The process of normalizing inputs
- D) A method to reduce the number of layers

Answer: **B) A regularization technique to penalize large weights**

---

**Q62. What is the main benefit of using the Swish activation function over ReLU?**

- A) Better handling of negative inputs
- B) Faster convergence
- C) Reduced computational complexity
- D) Improved gradient flow and higher accuracy

Answer: **D) Improved gradient flow and higher accuracy**

---

**Q63. Which of the following is a key advantage of using dilated convolutions in CNNs?**

- A) Increased receptive field without increasing the number of
- B) Faster training times
- C) Reduced overfitting
- D) Better gradient flow

Answer: **A) Increased receptive field without increasing the number of**

---

**Q64. What is the purpose of using spectral normalization in GANs?**

- A) To stabilize the training of the generator
- B) To normalize the input data
- C) To ensure the Lipschitz continuity of the discriminator
- D) To reduce the computational complexity

Answer: **C) To ensure the Lipschitz continuity of the discriminator**

---

**Q65. In the context of neural networks, what does the term "attention mechanism" refer to?**

- A) A method to initialize weights
- B) A technique to focus on relevant parts of the input data
- C) A regularization method to prevent overfitting
- D) A pooling operation to reduce dimensionality

Answer: **B) A technique to focus on relevant parts of the input data**

---

**Q66. Which loss function is commonly used for training GANs?**

- A) Mean Squared Error
- B) Cross-Entropy Loss
- C) Binary Cross-Entropy Loss
- D) Hinge Loss

Answer: **C) Binary Cross-Entropy Loss**

---

**Q67. What is the primary advantage of using capsule networks over traditional CNNs?**

- A) Reduced computational complexity
- B) Improved handling of spatial hierarchies and relationships
- C) Faster training times
- D) Better regularization

Answer: **B) Improved handling of spatial hierarchies and relationships**

---

**Q68. Which optimization technique utilizes a moving average of the squared gradients to scale the learning rate?**

- A) Adam
- B) Stochastic Gradient Descent
- C) RMSprop
- D) Adagrad

Answer: **C) RMSprop**

---

**Q69. In reinforcement learning, what does the term "policy gradient" refer to?**

- A) A method to update the value function
- B) A technique to compute the reward function
- C) An approach to optimize the policy directly by gradient ascent
- D) A way to initialize the policy network

Answer: **C) An approach to optimize the policy directly by gradient ascent**

---

**Q70. Which neural network architecture is specifically designed to handle graph-structured data?**

- A) Recurrent Neural Network (RNN)
- B) Convolutional Neural Network (CNN)
- C) Graph Neural Network (GNN)
- D) Generative Adversarial Network (GAN)

Answer: **C) Graph Neural Network (GNN)**

---

**Q71. What is the main purpose of using the ELU (Exponential Linear Unit) activation function in deep networks?**

- A) To improve gradient flow and prevent vanishing gradients
- B) To reduce computational complexity
- C) To enforce sparsity in the network
- D) To improve handling of negative inputs

Answer: **A) To improve gradient flow and prevent vanishing gradients**

---

**Q72. Which method is commonly used to initialize the weights of deep neural networks to avoid vanishing/exploding gradients?**

- A) Zeros initialization
- B) Random initialization
- C) Xavier (Glorot) initialization
- D) One-hot initialization

Answer: **C) Xavier (Glorot) initialization**

---

**Q73. In the context of DNNs, what is the function of the "encoder" in sequence-to-sequence models?**

- A) To generate new sequences from noise
- B) To encode input sequences into a fixed-length context vector
- C) To classify input sequences
- D) To normalize input data

Answer: **B) To encode input sequences into a fixed-length context vector**

---

## Generative AI

**Q74. What is the primary goal of generative AI?**

- A) To classify data into categories
- B) To generate new data that resembles existing data
- C) To optimize data retrieval
- D) To clean and preprocess data

Answer: **B) To generate new data that resembles existing data**

---

**Q75. Which of the following is a common generative model?**

- A) Convolutional Neural Network (CNN)
- B) Recurrent Neural Network (RNN)
- C) Generative Adversarial Network (GAN)
- D) Support Vector Machine (SVM)

Answer: **C) Generative Adversarial Network (GAN)**

---

**Q76. Who introduced Generative Adversarial Networks (GANs)?**

- A) Geoffrey Hinton
- B) Yann LeCun
- C) Ian Goodfellow
- D) Andrew Ng

Answer: **C) Ian Goodfellow**

---

**Q77. What are the two main components of a GAN?**

- A) Encoder and Decoder
- B) Generator and Discriminator
- C) Classifier and Regressor
- D) Trainer and Validator

Answer: **B) Generator and Discriminator**

---

**Q78. Which generative model is known for its ability to generate high-quality images?**

- A) Autoencoder
- B) GAN (Generative Adversarial Network)
- C) LSTM (Long Short-Term Memory)
- D) KNN (K-Nearest Neighbors)

Answer: **B) GAN (Generative Adversarial Network)**

---

**Q79. What does VAE stand for in the context of generative AI?**

- A) Variable Adversarial Encoder
- B) Variational Autoencoder
- C) Variable Autoregressive Estimator
- D) Variational Adversarial Estimator

Answer: **B) Variational Autoencoder**

---

**Q80. In the context of text generation, which model architecture is often used?**

- A) CNN
- B) RNN
- C) Transformer
- D) SVM

Answer: **C) Transformer**

---

**Q81. Which model is an example of a large-scale generative language model?**

- A) BERT
- B) ResNet
- C) GPT-3
- D) AlexNet

Answer: **C) GPT-3**

---

**Q82. Which of the following is a common application of generative AI?**

- A) Predicting stock prices
- B) Generating deepfake videos
- C) Image classification
- D) Data clustering

Answer: **B) Generating deepfake videos**

---

**Q83. What is a potential ethical concern related to generative AI?**

- A) It can improve data accuracy
- B) It can automate data cleaning
- C) It can create realistic fake content
- D) It can enhance image resolution

Answer: **C) It can create realistic fake content**

---

## Transformer Models

**Q84. What is the primary architectural innovation of the Transformer model?**

- A) Convolutional layers
- B) Recurrent layers
- C) Attention mechanism
- D) Dropout layers

Answer: **C) Attention mechanism**

---

**Q85. Who introduced the Transformer model in their seminal paper "Attention is All You Need"?**

- A) Geoffrey Hinton et al.
- B) Yann LeCun et al.
- C) Ian Goodfellow et al.
- D) Vaswani et al.

Answer: **D) Vaswani et al.**

---

**Q86. In the Transformer architecture, what does the self-attention mechanism primarily do?**

- A) Translates sentences from one language to another
- B) Allows the model to focus on different parts of the input sequence
- C) Reduces the dimensionality of the data
- D) Generates new data from noise

Answer: **B) Allows the model to focus on different parts of the input sequence**

---

**Q87. What is the purpose of the positional encoding in the Transformer model?**

- A) To add non-linearity to the model
- B) To provide information about the order of the input tokens
- C) To enhance the attention mechanism
- D) To normalize the input data

Answer: **B) To provide information about the order of the input tokens**

---

**Q88. Which component in the Transformer architecture is responsible for capturing dependencies between tokens regardless of their distance in the input sequence?**

- A) Feed-forward layers
- B) Convolutional layers
- C) Self-attention mechanism
- D) Pooling layers

Answer: **C) Self-attention mechanism**

---

**Q89. What is the purpose of the multi-head attention mechanism in the Transformer model?**

- A) To allow the model to focus on multiple parts of the sequence
- B) To reduce the computational complexity
- C) To increase the depth of the network
- D) To enhance gradient descent

Answer: **A) To allow the model to focus on multiple parts of the sequence**

---

**Q90. How is the Transformer architecture different from RNNs (Recurrent Neural Networks)?**

- A) It processes data sequentially
- B) It uses attention mechanisms to handle long-range dependencies
- C) It has a fixed size input and output
- D) It cannot handle variable-length sequences

Answer: **B) It uses attention mechanisms to handle long-range dependencies**

---

**Q91. Which of the following models is based on the Transformer architecture?**

- A) CNN
- B) LSTM
- C) BERT
- D) SVM

Answer: **C) BERT**

---

**Q92. In a Transformer, what is the function of the encoder?**

- A) To generate the final output sequence
- B) To process the input sequence and produce a set of representations
- C) To map the output sequence back to the input
- D) To apply convolutional operations on the input

Answer: **B) To process the input sequence and produce a set of representations**

---

**Q93. What is one of the main advantages of using the Transformer model over traditional sequence models like RNNs?**

- A) Transformers are simpler to implement
- B) Transformers do not require large datasets
- C) Transformers can process sequences in parallel
- D) Transformers have fewer parameters

Answer: **C) Transformers can process sequences in parallel**

---

## Convolutional Neural Networks (CNNs)

**Q94. What is the primary purpose of a convolutional layer in a CNN?**

- A) To reduce the dimensionality of the data
- B) To apply a series of filters to the input data, extracting
- C) To normalize the input data
- D) To perform linear transformations on the data

Answer: **B) To apply a series of filters to the input data, extracting**

---

**Q95. Which of the following is NOT a typical component of a CNN?**

- A) Convolutional layers
- B) Fully connected layers
- C) Recurrent layers
- D) Pooling layers

Answer: **C) Recurrent layers**

---

**Q96. What is the role of the activation function in a CNN?**

- A) To increase the learning rate of the network
- B) To introduce non-linearity into the network
- C) To reduce the number of parameters in the network
- D) To normalize the input data

Answer: **B) To introduce non-linearity into the network**

---

**Q97. What does the term "stride" refer to in the context of CNNs?**

- A) The number of filters in a convolutional layer
- B) The size of the pooling window
- C) The step size with which the filter moves across the input data
- D) The learning rate of the network

Answer: **C) The step size with which the filter moves across the input data**

---

**Q98. Which pooling operation is most commonly used in CNNs?**

- A) Mean pooling
- B) Max pooling
- C) Median pooling
- D) Min pooling

Answer: **B) Max pooling**

---

**Q99. In a CNN, what is the purpose of dropout layers?**

- A) To increase the learning rate
- B) To prevent overfitting by randomly deactivating a subset of
- C) To increase the number of filters in the convolutional layer
- D) To normalize the output of the network

Answer: **B) To prevent overfitting by randomly deactivating a subset of**

---

**Q100. What type of layer is typically used to flatten the output of convolutional layers before feeding it into fully connected layers?**

- A) Batch normalization layer
- B) Activation layer
- C) Flatten layer
- D) Pooling layer

Answer: **C) Flatten layer**

---

**Q101. What is the primary benefit of using convolutional layers over traditional fully connected layers in image processing tasks?**

- A) They are computationally more efficient and reduce the number of
- B) They increase the dimensionality of the data
- C) They are better suited for sequential data
- D) They require less preprocessing of the input data

Answer: **A) They are computationally more efficient and reduce the number of**

---

**Q102. Which type of data is CNNs particularly effective at processing?**

- A) Time series data
- B) Text data
- C) Image and video data
- D) Tabular data

Answer: **C) Image and video data**

---

**Q103. What is the main difference between 1D, 2D, and 3D convolutional layers?**

- A) The number of neurons in the layer
- B) The type of activation function used
- C) The dimensions of the input data they process (e.g., 1D for
- D) The size of the filters used in the layers

Answer: **C) The dimensions of the input data they process (e.g., 1D for**

---

## Convolutional Layers

**Q104. What is the primary purpose of a convolutional layer in a CNN?**

- A) To apply a linear transformation to the input data
- B) To extract local features from the input data using filters
- C) To reduce the dimensionality of the input data
- D) To normalize the input data

Answer: **B) To extract local features from the input data using filters**

---

**Q105. In a convolutional layer, what does a filter (or kernel) do?**

- A) Normalizes the input data
- B) Applies an activation function to the input data
- C) Slides over the input data and computes dot products
- D) Reduces the size of the input data

Answer: **C) Slides over the input data and computes dot products**

---

**Q106. What does the term "stride" refer to in the context of a convolutional layer?**

- A) The number of filters in the layer
- B) The size of the filter
- C) The step size with which the filter moves across the input data
- D) The number of input channels

Answer: **C) The step size with which the filter moves across the input data**

---

**Q107. Which of the following effects does increasing the stride in a convolutional layer have?**

- A) It increases the output size
- B) It decreases the output size
- C) It has no effect on the output size
- D) It normalizes the output data

Answer: **B) It decreases the output size**

---

**Q108. What is the role of padding in a convolutional layer?**

- A) To reduce the number of parameters
- B) To prevent overfitting
- C) To maintain the spatial dimensions of the input data
- D) To increase the depth of the network

Answer: **C) To maintain the spatial dimensions of the input data**

---

**Q109. What is the common default value for stride in a convolutional layer if not specified?**

- A) 0
- B) 1
- C) 2
- D) 3

Answer: **B) 1**

---

**Q110. How does a convolutional layer handle multiple input channels, such as in RGB images?**

- A) It converts them to a single channel
- B) It applies the same filter to each channel separately
- C) It uses a separate filter for each channel and sums the results
- D) It ignores all but one channel

Answer: **C) It uses a separate filter for each channel and sums the results**

---

**Q111. What happens to the output feature map size if you use 'valid' padding in a convolutional layer?**

- A) It increases the size of the feature map
- B) It decreases the size of the feature map
- C) It keeps the size of the feature map the same
- D) It doubles the size of the feature map

Answer: **B) It decreases the size of the feature map**

---

**Q112. What is the main advantage of using smaller filters (e.g., 3x3) in a convolutional layer?**

- A) They are computationally less expensive and capture fine details
- B) They require less memory
- C) They are easier to implement
- D) They increase the output size

Answer: **A) They are computationally less expensive and capture fine details**

---

**Q113. Which of the following statements is true about the convolution operation in a convolutional layer?**

- A) It is linear and does not involve any non-linear activation
- B) It is non-linear and always involves an activation function
- C) It involves element-wise multiplication followed by a summation
- D) It involves calculating the gradient of the input data

Answer: **C) It involves element-wise multiplication followed by a summation**

---

## Pooling Layers

**Q114. What is the primary purpose of a pooling layer in a CNN?**

- A) To apply a linear transformation to the input data
- B) To reduce the spatial dimensions (width and height) of the input
- C) To increase the number of parameters in the network
- D) To normalize the input data

Answer: **B) To reduce the spatial dimensions (width and height) of the input**

---

**Q115. Which of the following is the most commonly used pooling operation in CNNs?**

- A) Mean pooling
- B) Max pooling
- C) Min pooling
- D) Sum pooling

Answer: **B) Max pooling**

---

**Q116. What does max pooling do?**

- A) It calculates the average of all values in the pooling window
- B) It selects the maximum value from the pooling window
- C) It selects the minimum value from the pooling window
- D) It sums all values in the pooling window

Answer: **B) It selects the maximum value from the pooling window**

---

**Q117. What effect does pooling have on the computational complexity of the network?**

- A) It increases the computational complexity
- B) It has no effect on the computational complexity
- C) It reduces the computational complexity
- D) It normalizes the computational complexity

Answer: **C) It reduces the computational complexity**

---

**Q118. Which of the following pooling methods can be used to reduce overfitting?**

- A) Mean pooling
- B) Max pooling
- C) Stochastic pooling
- D) Global pooling

Answer: **C) Stochastic pooling**

---

**Q119. What is the typical size of a pooling window in a pooling layer?**

- A) 1x1
- B) 2x2
- C) 3x3
- D) 4x4

Answer: **B) 2x2**

---

**Q120. What does the term "stride" refer to in the context of a pooling layer?**

- A) The number of filters used in the pooling layer
- B) The size of the pooling window
- C) The step size with which the pooling window moves across the input
- D) The depth of the pooling layer

Answer: **C) The step size with which the pooling window moves across the input**

---

**Q121. What is the difference between max pooling and average pooling?**

- A) Max pooling selects the maximum value in the pooling window, while
- B) Max pooling calculates the average of the values, while average
- C) Max pooling and average pooling perform the same operation
- D) Max pooling selects the minimum value in the pooling window, while

Answer: **A) Max pooling selects the maximum value in the pooling window, while**

---

**Q122. What is global pooling?**

- A) Pooling applied to the entire input data
- B) Pooling that uses the maximum value across all channels
- C) Pooling that normalizes the data
- D) Pooling that is applied at every layer of the network

Answer: **A) Pooling applied to the entire input data**

---

**Q123. Which of the following is NOT an advantage of using pooling layers?**

- A) Reducing the spatial dimensions of the input
- B) Reducing the number of parameters and computations in the network
- C) Making the model more invariant to small translations of the input
- D) Increasing the resolution of the input data

Answer: **D) Increasing the resolution of the input data**

---


---


---

# ═══ ASSET 2: 30 Basic MCQs on Gradient Descent (30_Basic_MCQs_on_Gradient_Descent.docx) ═══
*(Everything above this line came from Sample_MCQs_DNN.docx — Asset 1. Everything below is from the newly uploaded Gradient Descent MCQ set — Asset 2.)*

---

## Contents — Asset 2

- Concept & Definition (3 questions)
- Learning Rate (4 questions)
- Gradient and Direction (3 questions)
- Cost Function (4 questions)
- Convergence (3 questions)
- Variants of Gradient Descent (3 questions)
- Backpropagation & Gradients (4 questions)
- Practical Issues (4 questions)
- Edge Cases (2 questions)

---


## Concept & Definition

**Q124. What is the primary goal of gradient descent?**

- A) To find the global maximum of the cost function
- B) To find the minimum of the cost function
- C) To compute the loss for a single sample
- D) To update the learning rate automatically

Answer: **B) To find the minimum of the cost function**

---

**Q125. The gradient of a function points in which direction?**

- A) Towards the minimum
- B) Towards the maximum (steepest increase)
- C) Perpendicular to the surface
- D) In a random direction

Answer: **B) Towards the maximum (steepest increase)** — The gradient points in the direction of steepest *increase*, so we step in the *negative* gradient direction to go downhill.

---

**Q126. In the update rule w_new = w_old − α × ∇L, what does the minus sign represent?**

- A) The learning rate should be negative
- B) We want to move in the opposite direction of the gradient
- C) The cost function is decreasing
- D) The weight is being reset

Answer: **B) We want to move in the opposite direction of the gradient** — The minus sign ensures we step *downhill* (opposite to the steepest increase).

---

## Learning Rate

**Q127. What happens if the learning rate α is too large?**

- A) The algorithm converges faster
- B) The algorithm may overshoot the minimum and diverge
- C) The gradient becomes larger
- D) The number of iterations increases slightly

Answer: **B) The algorithm may overshoot the minimum and diverge** — Large steps can skip over the minimum or bounce back and forth instead of settling.

---

**Q128. What happens if the learning rate α is too small?**

- A) The algorithm converges instantly
- B) The algorithm diverges
- C) The algorithm converges very slowly (many iterations needed)
- D) The gradient becomes zero

Answer: **C) The algorithm converges very slowly (many iterations needed)** — Tiny steps mean tiny progress per iteration, so it takes many iterations to reach the minimum.

---

**Q129. Which learning rate typically works best?**

- A) 0.0001 for all problems
- B) 1.0 for all problems
- C) A value that balances convergence speed and stability (varies by     problem)
- D) The learning rate doesn't matter

Answer: **C) A value that balances convergence speed and stability (varies by     problem)** — The optimal learning rate depends on the problem, network architecture, and dataset. Common values range from 0.001 to 0.1.

---

**Q130. In gradient descent, if the loss stops decreasing after many iterations, what is most likely?**

- A) The algorithm has converged (reached a minimum)
- B) The learning rate is too large
- C) The gradient is zero (or very close to zero)
- D) The cost function is broken

Answer: **C) The gradient is zero (or very close to zero)** — When ∇L ≈ 0, the weight updates become negligible: w_new ≈ w_old − α × 0 = w_old.

---

## Gradient and Direction

**Q131. If the gradient ∂L/∂w is positive, what does that mean?**

- A) Increasing w will decrease the loss
- B) Increasing w will increase the loss
- C) The weight is at a minimum
- D) The weight should not be updated

Answer: **B) Increasing w will increase the loss** — A positive gradient means the function is increasing in that direction, so we step in the negative direction (decrease w).

---

**Q132. If the gradient ∂L/∂w is negative, what should we do?**

- A) Decrease w further (move left on the curve)
- B) Increase w (move right, towards lower loss)
- C) Keep w constant
- D) Double the learning rate

Answer: **B) Increase w (move right, towards lower loss)** — Negative gradient means increasing w will decrease the loss. The update w_new = w_old − α × (negative) = w_old + α × (positive) increases w.

---

**Q133. What does it mean when ∇L = 0?**

- A) The algorithm should stop immediately
- B) We have reached a critical point (could be a minimum, maximum,     or saddle point)
- C) The cost function is undefined
- D) The learning rate is zero

Answer: **B) We have reached a critical point (could be a minimum, maximum,     or saddle point)** — Zero gradient indicates a stationary point, which could be a local minimum, global minimum, or saddle point.

---

## Cost Function

**Q134. In gradient descent, the cost function L measures:**

- A) The number of training samples
- B) The difference between predicted and actual values
- C) The learning rate
- D) The number of iterations

Answer: **B) The difference between predicted and actual values** — The cost function (or loss function) quantifies prediction error.

---

**Q135. Which cost function is most commonly used for regression?**

- A) Cross-entropy loss
- B) Mean squared error (MSE)
- C) Accuracy
- D) Precision

Answer: **B) Mean squared error (MSE)** — MSE = (1/N) × Σ(predicted − actual)² is standard for regression.

---

**Q136. Why is a convex cost function preferable in gradient descent?**

- A) It requires fewer parameters
- B) It converges faster and any local minimum is also the global     minimum
- C) It doesn't need a learning rate
- D) It automatically adjusts weights

Answer: **B) It converges faster and any local minimum is also the global     minimum** — Convex functions have a single global minimum, so gradient descent will find it regardless of starting point.

---

**Q137. If the cost function is non-convex (has multiple local minima), what might happen?**

- A) Gradient descent is impossible
- B) Gradient descent may get stuck in a local minimum, not the     global minimum
- C) The learning rate automatically adjusts
- D) The algorithm always finds the global minimum

Answer: **B) Gradient descent may get stuck in a local minimum, not the     global minimum** — Non-convex functions can trap gradient descent in suboptimal local minima.

---

## Convergence

**Q138. What does convergence in gradient descent mean?**

- A) The loss reaches exactly zero
- B) The loss stops decreasing significantly (gradient approaches     zero)
- C) The algorithm runs for a fixed number of iterations
- D) All weights become zero

Answer: **B) The loss stops decreasing significantly (gradient approaches     zero)** — Convergence means the algorithm has reached a point where further updates produce negligible improvement.

---

**Q139. How many iterations does gradient descent typically need?**

- A) Always 100 iterations
- B) Always 1000 iterations
- C) Depends on the problem, learning rate, and starting point
- D) Exactly as many as there are training samples

Answer: **C) Depends on the problem, learning rate, and starting point** — Convergence speed varies widely based on the landscape and hyperparameters.

---

**Q140. Which statement about batch gradient descent is true?**

- A) It updates weights using one sample at a time
- B) It uses the entire training dataset to compute one gradient per     iteration
- C) It uses random subsets of data
- D) It doesn't require a learning rate

Answer: **B) It uses the entire training dataset to compute one gradient per     iteration** — Batch GD computes ∇L across all training samples before a single weight update.

---

## Variants of Gradient Descent

**Q141. What is stochastic gradient descent (SGD)?**

- A) Gradient descent with random noise added
- B) Gradient descent using one sample at a time
- C) Gradient descent using the entire dataset
- D) Gradient descent with multiple learning rates

Answer: **B) Gradient descent using one sample at a time** — SGD updates weights using the gradient from a single sample, making updates frequent but noisier.

---

**Q142. What is mini-batch gradient descent?**

- A) Using one sample per update
- B) Using all samples per update
- C) Using a small batch (e.g., 32 samples) per update
- D) Using random batches of varying sizes

Answer: **C) Using a small batch (e.g., 32 samples) per update** — Mini-batch GD balances speed and stability by computing the gradient over a small batch.

---

**Q143. Which variant typically converges fastest in practice?**

- A) Batch gradient descent
- B) Stochastic gradient descent
- C) Mini-batch gradient descent
- D) All converge at the same rate

Answer: **C) Mini-batch gradient descent** — Mini-batch GD balances the stability of batch GD with the speed of SGD.

---

## Backpropagation & Gradients

**Q144. What is backpropagation?**

- A) Moving weights backward in the network
- B) The process of computing gradients by working backward through     the network using the chain rule
- C) Restarting the algorithm from scratch
- D) Reversing the order of layers

Answer: **B) The process of computing gradients by working backward through     the network using the chain rule** — Backpropagation systematically computes ∂L/∂w for every weight using the chain rule.

---

**Q145. In backpropagation, the chain rule is used to:**

- A) Chain weights together
- B) Multiply local gradients to compute gradients of early layers
- C) Speed up training
- D) Prevent overfitting

Answer: **B) Multiply local gradients to compute gradients of early layers** — The chain rule propagates gradients backward: ∂L/∂w = ∂L/∂output × ∂output/∂hidden × ∂hidden/∂w.

---

**Q146. What is the vanishing gradient problem?**

- A) Gradients become too large during backpropagation
- B) Gradients shrink exponentially as they travel backward through     many layers, causing early layers to learn slowly
- C) The learning rate vanishes
- D) Weights disappear during training

Answer: **B) Gradients shrink exponentially as they travel backward through     many layers, causing early layers to learn slowly** — Deep networks with sigmoid activations multiply many derivatives < 1 together, resulting in tiny gradients for early layers.

---

**Q147. Which activation function helps prevent vanishing gradients?**

- A) Sigmoid
- B) Tanh
- C) ReLU
- D) All equally well

Answer: **C) ReLU** — ReLU has derivative = 1 (when x \> 0), preserving gradient magnitude through layers.

---

## Practical Issues

**Q148. What is overfitting in the context of gradient descent?**

- A) The learning rate is too high
- B) The model memorizes training data and performs poorly on unseen     data
- C) The cost function is non-convex
- D) The algorithm converges too slowly

Answer: **B) The model memorizes training data and performs poorly on unseen     data** — Overfitting occurs when a model learns noise in the training data rather than general patterns.

---

**Q149. How can regularization help gradient descent?**

- A) It increases the learning rate
- B) It penalizes large weights, encouraging simpler models and     reducing overfitting
- C) It speeds up convergence
- D) It eliminates the need for a learning rate

Answer: **B) It penalizes large weights, encouraging simpler models and     reducing overfitting** — L1/L2 regularization adds a penalty term to the loss: L_total = L + λ × (penalty on weights).

---

**Q150. What does momentum in gradient descent do?**

- A) Increases the learning rate
- B) Accumulates past gradients to accelerate convergence and smooth     noisy updates
- C) Resets weights periodically
- D) Prevents overfitting

Answer: **B) Accumulates past gradients to accelerate convergence and smooth     noisy updates** — Momentum maintains a velocity term v = β×v + ∇L, causing the algorithm to build up speed in consistent directions.

---

**Q151. In adaptive optimizers like Adam, what is adapted?**

- A) The network architecture
- B) The learning rate for each parameter individually based on     gradient history
- C) The cost function
- D) The batch size

Answer: **B) The learning rate for each parameter individually based on     gradient history** — Adam (and RMSProp) compute per-parameter learning rates, giving frequently-updated parameters smaller steps and rarely-updated ones larger steps.

---

## Edge Cases

**Q152. If all gradients ∂L/∂w are exactly zero after the first iteration, what likely happened?**

- A) The algorithm has converged successfully
- B) The weights were initialized to a saddle point or critical point
- C) The learning rate is too small
- D) The cost function is broken

Answer: **B) The weights were initialized to a saddle point or critical point** — Starting exactly at a critical point (rare but possible) means no gradient information to update weights.

---

**Q153. Why is weight initialization important in gradient descent?**

- A) Poor initialization can trap the algorithm in bad local minima     or saddle points
- B) Initialization doesn't matter; gradient descent works     regardless
- C) It only affects the learning rate
- D) It guarantees convergence to the global minimum

Answer: **A) Poor initialization can trap the algorithm in bad local minima     or saddle points** — Good initialization places the algorithm in a region of the loss landscape where gradient descent can find good solutions efficiently.

---


---

# ═══ ASSET 3: Basic MCQs on RNN (Basic_MCQs_on_RNN.docx) ═══
*(Everything above this line came from Assets 1–2. Everything below is from the newly uploaded RNN MCQ set — Asset 3.)*

**Note:** The source document contains two separate 20-question MCQ sets on RNN. In Set 2, question 19 was presented only as an embedded image (not extractable text) and has been omitted here rather than guessed. The document also included Very Short/Short/Long Answer and Viva sections (open-ended, not MCQs) and a "Numericals practice" section (calculation-based) — both skipped, consistent with your no-code/no-calculation preference.

---

## Contents — Asset 3

- RNN — Set 1 (20 questions)
- RNN — Set 2 (19 questions — Q19 omitted, image-based in source)

---


## RNN — Set 1

**Q154. What does RNN stand for?**

- A) Recursive Neural Network
- B) Recurrent Neural Network
- C) Regression Neural Network
- D) Random Neural Network

Answer: **B) Recurrent Neural Network**

---

**Q155. RNNs are primarily designed for processing:**

- A) Tabular data
- B) Sequential data
- C) Image pixels only
- D) Graph data only

Answer: **B) Sequential data**

---

**Q156. Which of the following is an example of sequential data?**

- A) Image
- B) Video frames
- C) Audio signal
- D) Both B and C

Answer: **D) Both B and C**

---

**Q157. The main advantage of an RNN over a traditional neural network is:**

- A) Faster computation
- B) Ability to remember previous information
- C) Requires fewer parameters
- D) Needs no training

Answer: **B) Ability to remember previous information**

---

**Q158. In an RNN, information is stored in:**

- A) Output layer
- B) Hidden state
- C) Input layer
- D) Bias

Answer: **B) Hidden state**

---

**Q159. Which application commonly uses RNNs?**

- A) Image Classification
- B) Sentiment Analysis
- C) Face Detection
- D) Object Detection

Answer: **B) Sentiment Analysis**

---

**Q160. Which architecture is commonly used for sentiment analysis?**

- A) One-to-One
- B) One-to-Many
- C) Many-to-One
- D) Many-to-Many

Answer: **C) Many-to-One**

---

**Q161. Which architecture is used in image captioning?**

- A) One-to-One
- B) One-to-Many
- C) Many-to-One
- D) Many-to-Many

Answer: **B) One-to-Many**

---

**Q162. Machine translation is an example of:**

- A) One-to-One
- B) One-to-Many
- C) Many-to-One
- D) Many-to-Many

Answer: **D) Many-to-Many**

---

**Q163. Which problem is commonly faced by Simple RNNs?**

- A) Overfitting only
- B) Underfitting only
- C) Vanishing Gradient
- D) Class imbalance

Answer: **C) Vanishing Gradient**

---

**Q164. Which RNN variant is designed to overcome the vanishing gradient**

- A) CNN
- B) LSTM
- C) Perceptron
- D) Autoencoder

Answer: **B) LSTM**

---

**Q165. GRU stands for:**

- A) General Recurrent Unit
- B) Gated Recurrent Unit
- C) Global Recurrent Unit
- D) Gradient Recurrent Unit

Answer: **B) Gated Recurrent Unit**

---

**Q166. Compared to LSTM, GRU has:**

- A) More gates
- B) Fewer gates
- C) No gates
- D) Five gates

Answer: **B) Fewer gates**

---

**Q167. Which of the following is NOT an application of RNN?**

- A) Speech Recognition
- B) Language Translation
- C) Time-Series Prediction
- D) Sorting an array

Answer: **D) Sorting an array**

---

**Q168. In an RNN, the hidden state at time t depends on:**

- A) Current input only
- B) Previous hidden state only
- C) Current input and previous hidden state
- D) Output only

Answer: **C) Current input and previous hidden state**

---

**Q169. Which type of neural network is most suitable for predicting stock**

- A) CNN
- B) RNN
- C) GAN
- D) Autoencoder

Answer: **B) RNN**

---

**Q170. Which of the following is true about Simple RNN?**

- A) It remembers long-term dependencies perfectly.
- B) It has no hidden state.
- C) It struggles with long sequences.
- D) It has three gates.

Answer: **C) It struggles with long sequences.**

---

**Q171. Which gate is NOT present in a GRU?**

- A) Update Gate
- B) Reset Gate
- C) Output Gate
- D) None of the above

Answer: **C) Output Gate**

---

**Q172. Which gate decides what information should be forgotten in an LSTM?**

- A) Input Gate
- B) Forget Gate
- C) Output Gate
- D) Reset Gate

Answer: **B) Forget Gate**

---

**Q173. Modern NLP models like ChatGPT are mainly based on:**

- A) Simple RNN
- B) LSTM
- C) GRU
- D) Transformer

Answer: **D) Transformer**

---


## RNN — Set 2

**Q174. What is the primary purpose of a Recurrent Neural Network (RNN)?**

- A) To process tabular data
- B) To process sequential data
- C) To compress images
- D) To cluster data

Answer: **B) To process sequential data**

---

**Q175. Which of the following data types is best suited for an RNN?**

- A) Images
- B) Audio signals
- C) Tabular data
- D) Static graphs

Answer: **B) Audio signals**

---

**Q176. An RNN differs from a feedforward neural network because it:**

- A) Has no hidden layer
- B) Contains recurrent (feedback) connections
- C) Has only one neuron
- D) Does not require training

Answer: **B) Contains recurrent (feedback) connections**

---

**Q177. The hidden state in an RNN is used to:**

- A) Store the learning rate
- B) Store previous information
- C) Store only the output
- D) Store the bias values

Answer: **B) Store previous information**

---

**Q178. Which of the following is the main feature of an RNN?**

- A) Parallel computation
- B) Memory of previous inputs
- C) Image segmentation
- D) Feature scaling

Answer: **B) Memory of previous inputs**

---

**Q179. In an RNN, the hidden state at time t depends on:**

- A) Only the current input
- B) Only the previous hidden state
- C) The current input and previous hidden state
- D) The output only

Answer: **C) The current input and previous hidden state**

---

**Q180. Which symbol generally represents the hidden state in an RNN?**

- A) xₜ
- B) hₜ
- C) yₜ
- D) wₜ

Answer: **B) hₜ**

---

**Q181. Which symbol usually represents the input at time step t?**

- A) hₜ
- B) yₜ
- C) xₜ
- D) bₜ

Answer: **C) xₜ**

---

**Q182. The output at time step t is generally denoted by:**

- A) yₜ
- B) hₜ
- C) xₜ
- D) wₜ

Answer: **A) yₜ**

---

**Q183. Which of the following is a common application of Basic RNN?**

- A) Weather forecasting
- B) Sentiment analysis
- C) Speech recognition
- D) All of the above

Answer: **D) All of the above**

---

**Q184. In an RNN, the same weights are:**

- A) Different for every time step
- B) Shared across all time steps
- C) Used only once
- D) Randomly initialized at every step

Answer: **B) Shared across all time steps**

---

**Q185. Which training algorithm is commonly used for RNNs?**

- A) Gradient Descent
- B) K-Means
- C) Backpropagation Through Time (BPTT)
- D) Decision Tree

Answer: **C) Backpropagation Through Time (BPTT)**

---

**Q186. A major limitation of a Basic RNN is:**

- A) High memory usage only
- B) Vanishing gradient problem
- C) Cannot process sequences
- D) Cannot use activation functions

Answer: **B) Vanishing gradient problem**

---

**Q187. Which activation function is commonly used in the hidden layer of a**

- A) Softmax
- B) Sigmoid or Tanh
- C) Linear
- D) Identity

Answer: **B) Sigmoid or Tanh**

---

**Q188. In a Many-to-One RNN architecture, the network produces:**

- A) One output for every input
- B) Multiple outputs from one input
- C) One output after processing the entire input sequence
- D) No output

Answer: **C) One output after processing the entire input sequence**

---

**Q189. Which architecture is suitable for sentiment analysis?**

- A) One-to-One
- B) One-to-Many
- C) Many-to-One
- D) Many-to-Many

Answer: **C) Many-to-One**

---

**Q190. Which of the following best describes the flow of information in an**

- A) Only forward between layers
- B) Forward with recurrent connections through time
- C) Backward only
- D) Randomly connected

Answer: **B) Forward with recurrent connections through time**

---

**Q191. Why is an RNN considered suitable for sequence modeling?**

- A) It ignores previous inputs.
- B) It processes all inputs independently.
- C) It maintains a hidden state that captures context.
- D) It only works with images.

Answer: **C) It maintains a hidden state that captures context.**

---

**Q192. Which statement about Basic RNNs is TRUE?**

- A) They cannot process variable-length sequences.
- B) They are designed for sequential data.
- C) They completely eliminate the vanishing gradient problem.
- D) They do not use hidden states.

Answer: **B) They are designed for sequential data.**

---
