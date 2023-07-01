# Part 2: [Advanced Learning Algorithms](2.%20Advanced%20Learning%20Algorithms/)
### <ul>Week 1: Neural Networks</ul>
#### Neural Network Intuition
* __Neural Networks__ are a type of machine learning algorithm that is inspired by the human brain.
* In the brain, some neurons are connected. The input of one neuron is the output of another neuron.
  * We know very little about the brain, thus we do not mimic the brain exactly. Instead, we use a simplified model of the brain.
* Neural networks have taken off in the last few years because of the availability of large datasets and the availability of powerful computers, which allows us to train neural networks.

#### Neural Network Model
* __Activations__ are the output of a neuron.
  * To calculate the activation of a neuron, we take the weighted sum of the inputs, add the bias, and then apply the activation function.
  * The formula Andrew uses is $a^{[l]} = g^{[l]}(w^{[l]} * a^{[l-1]} + b^{[l]})$ where $a^{[l]}$ is the activation of the lth layer, $g^{[l]}$ is the activation function of the lth layer, $w^{[l]}$ is the weight matrix of the lth layer, $a^{[l-1]}$ is the activation of the (l-1)th layer, and $b^{[l]}$ is the bias vector of the lth layer.
* Calculating the activation of a neuron is done with the __forward propagation__ algorithm.
  
#### Artificial General Intelligence
* __Artificial General Intelligence__ is when a machine can perform any intellectual task that a human can perform.
  * This is a very ambitious goal.
  * The goal of __artificial narrow intelligence__ is to create a model that can perform a specific task, but this is not AGI.
    * For example, a model that can recognize handwritten digits.
  * Andrew thinks it will take more than a decade to achieve this, and describes possible paths to it.

### <ul>Week 2: Neural Network Training</ul>

#### Neural Net Training
* Training a neural net follows these 3 steps:
  1. Define the neural network structure (number of input units, number of hidden units, etc).
  2. Initialize the model's parameters.
  3. Loop:
     1. Implement forward propagation.
     2. Compute loss.
     3. Implement backward propagation to get the gradients.
     4. Update parameters (gradient descent).
* Today some libraries can do all of this for you, but it is still good to understand how it works.

#### Activation Functions
* Some alternatives to the sigmoid function are the __tanh__ function and the __ReLU__ function.
  * The tanh function is defined as $tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$.
  * The ReLU function is defined as $ReLU(x) = max(0, x)$.
* Choosing an activation function is an important part of building a neural network.
  * The sigmoid function is good for binary classification.
  * The tanh function is good for binary classification and hidden layers.
  * The ReLU function is good for hidden layers.
* We need activation functions that are not linear, because if they were linear, then the neural network would be equivalent to a linear regression model, which defeats the purpose of using a neural network.

#### Multiclass Classification
* __Multiclass classification__ is when there are more than 2 classes.
  * For example, classifying handwritten digits.
* __Softmax__ is a function that is used to calculate the probability of an input belonging to a certain class.
  * The softmax function is defined as $softmax(x) = \frac{e^x}{\sum_{i=1}^{n} e^x}$.
  * The softmax function is used to calculate the probability of an input belonging to a certain class.
  * The softmax function is used in multiclass classification.
* __Cross-entropy__ is a function that is used to calculate the loss of a neural network that does multiclass classification.
  * The cross-entropy function is defined as $H(y, \hat{y}) = -\sum_{i=1}^{n} y_i \log(\hat{y}_i)$.
  * The cross-entropy function is used in conjunction with the softmax function.
* __MNIST__ is a large database of handwritten digits that is used to train models to recognize handwritten digits.
  * The MNIST database contains 60,000 training examples and 10,000 test examples.
  * Each example is a 28x28 pixel grayscale image of a handwritten digit.
  * Each pixel is represented by an integer between 0 and 255.
  * Each image also has a label, which is an integer between 0 and 9, inclusive, representing which digit is in the image.
* __Logits__ are the raw outputs of a neural network.
  * The logits are the inputs to the softmax function.
  * The logits are the outputs of the last layer of a neural network.

#### Optimization
* __Adam__ is an optimization algorithm that is used to update the parameters of a neural network and stands for __adaptive moment estimation__.

### <ul>Week 3: Advice for Applying Machine Learning</ul>

#### Advice for Applying Machine Learning
* We have the tools to build a machine-learning model, but how do we know if it will work? There are a few ways we could improve our model:
  * Get more training data.
  * Try smaller sets of features.
  * Try getting additional features.
  * Try adding polynomial features.
  * Try decreasing $\lambda$.
  * Try increasing $\lambda$.
* Choosing well means saving a lot of time that would otherwise be wasted, but choosing correctly could be tricky.
* To evaluate the model, we can look at the fraction of misclassifications on the training set and test set.
  * If the training set error is low, but the test set error is high, then the model is overfitting.
  * If the training set error is high, then the model is underfitting.
  * If the training set error is low, and the test set error is low, then the model is working well.
  * To do this evaluation we split data into a training set, test set, and cross-validation set.
    * The training set is used to train the model.
    * The test set is used to evaluate the model.
    * The cross-validation set is used to choose the model.

#### Bias and Variance
* High bias means the model is underfitting, which means the model is too simple.
* High variance means the model is overfitting, which means the model is too complex.
* To establish a baseline we can look at human-level performance, or look at the performance of current models if there are any.

#### Machine Learning Development Process
* The iterative loop Andrew suggests:
  1. Choose an architecture.
  2. Train the model.
  3. Evaluate the model.
  4. Analyze the results.
  5. Iterate.
* Error analysis is the process of looking at the misclassifications and trying to understand why the model is making those mistakes. By categorizing misclassifications, we can try to understand what the model is doing wrong and try to fix it and look at where to focus our efforts.
* Adding more data is tempting, but it is not always the best solution. A better approach might be to evaluate which _type_ of data is necessary and then try to get more of that type of data. This could boost performance more than just adding more data.
  * One way to add more data would be to augment the data we already have. __Data augmentation__ is the process of creating new data from the data we already have.
    * For example, if we have a dataset of images of cats, we could rotate the images, flip them, and change the brightness of the images to create new images.
* It might be better to take a data-centric approach to machine learning, compared to a conventional model-centric approach.
  * __Transfer Learning__ is the process of using a model that has already been trained on a different dataset to train a model on a new dataset.
    * For example, we could use a model that has already been trained on the ImageNet dataset to train a model on the MNIST dataset.
  * To apply transfer learning we would take a trained model and replace the last layer with a new layer that is specific to the new dataset.
* Full cycle of machine learning project:
  1. Define the problem.
  2. Collect the data.
  3. Prepare the data.
  4. Choose a model.
  5. Train the model.
  6. Evaluate the model.
  7. Analyze the results.
  8. Deploy the model.
  9. Monitor the model.
* Machine Learning is affecting billions of people and it is important to make it fair and ethical by making them unbiased and transparent.
  * Sometimes these models become viral due to their engagement and popularity, but they can also be harmful.
* Some guidelines Andrew has:
  1. Have a more diverse team and emphasize problems that might harm minority groups.
  2. Audit systems against possible harm before deployment.
  3. Develop a mitigation strategy.

### <ul>Week 4: Decision tree model</ul>

#### Decision Trees
* __Decision trees__ are a type of supervised learning algorithm that can be used for both classification and regression.
* Some decisions include:
  1. How to choose what feature to split on at each node?
    - We need to maximize purity.
  2. When do you stop splitting?
    - When the node is 100% pure, or when we reach the maximum depth of the tree.

#### Decision Tree Learning
* __Entropy__ is a measure of impurity.
  * The entropy of a node is defined as $H(S) = -\sum_{i=1}^{n} p_i \log_2(p_i)$.
  * The entropy of a node is 0 if the node is pure.
  * The entropy of a node is 1 if the node is equally likely to be any of the classes.
* Process:
  1. Calculate information gain for each feature.
  2. Split on the feature with the highest information gain and create left and right branches.
  3. Repeat until a stopping condition is met.
  - We can do this with a recursive algorithm.
* We can use one hot encoding on categorical variables to choose features. __One hot encoding__ is the process of representing categorical variables as binary vectors.
  * For example, if we have a categorical variable with 3 possible values, we could represent it as a vector of length 3, where the first element is 1 if the value is the first value, the second element is 1 if the value is the second value, and the third element is 1 if the value is the third value.
* To split continuous variables we could use information gain to choose the best split point.

#### Tree Ensembles
* __Tree ensembles__ are many decision trees combined together to make a more powerful model.
* __Sampling with replacement__ is the process of randomly sampling from a dataset with replacement, which means that the same example can be sampled multiple times.
* A popular ensembling algorithm is __random forests__, which is a tree ensemble algorithm that uses decision trees as the base learner.
  * Random forests are a type of bagging algorithm.
  * __Bagging__ is the process of training many models on different subsets of the data and then combining the results.
  * Random means that we randomly sample the features at each split.
* __XGBoost__ is a tree ensemble algorithm that uses decision trees as the base learner.
  * XGBoost stands for __extreme gradient boosting__.
  * __Boosting__ is the process of training many models sequentially, where each model tries to correct the mistakes of the previous model.
* Using tree ensembles rather than neural networks has some advantages and disadvantages:
  * Advantages:
    - They are easier to interpret.
    - They are faster to train.
    - They are less prone to overfitting.
  * Disadvantages:
    - They are less flexible.
    - They are less likely to perform well on complex tasks.
    - They are less likely to perform well on large datasets.

<hr />

## Part 3: [Unsupervised Learning, Recommenders, Reinforcement Learning](./3.%20Unsupervised%20Learning%2C%20Recommenders%2C%20Reinforcement%20Learning/)
### <ul>Week 1: Welcome!</ul>
#### Clustering
* __Clustering__ is the process of grouping data points into clusters.
* The __K-means algorithm__ is an algorithm that can be used to cluster data.
  * The algorithm works by:
    1. Randomly initialize K cluster centroids.
    2. Assign each data point to the closest cluster centroid.
    3. Update the cluster centroids to be the mean of the data points in the cluster.
    4. Repeat steps 2 and 3 until the cluster centroids no longer change.
  * The optimization objective for k-means is to minimize the sum of the squared distances between each data point and its cluster centroid.

#### Anomoly Detection
* __Anomaly detection__ is the process of identifying data points that are different from the rest of the data.
* A __Gausian distribution__ is a distribution that is shaped like a bell curve.
  * The mean of the distribution is the center of the bell curve.
  * The variance of the distribution is the width of the bell curve.
  * Also known as the "Normal distribution".
  * The distribution is defined as $p(x) = \frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}$.
    * To estimate the mean and the variance we use the maximum likelihood estimate.
      * mu is estimatied by $\mu = \frac{1}{m}\sum_{i=1}^{m}x^{(i)}$.
      * variance is estimated by $\sigma^2 = \frac{1}{m}\sum_{i=1}^{m}(x^{(i)}-\mu)^2$.
  * If distributions are not gaussian you can log transform the data to make it more gaussian.

### <ul>Week 2: Recommender Systems</ul>
#### Collaborative Filtering
* __Collaborative filtering__ is a type of recommender system that makes predictions based on the past behavior of similar users.
  * The formula for the cost function used in collaborative filtering is $J(x^{(1)},...,x^{(n_m)},\theta^{(1)},...,\theta^{(n_u)}) = \frac{1}{2}\sum_{(i,j):r(i,j)=1}((\theta^{(j)})^Tx^{(i)}-y^{(i,j)})^2 + \frac{\lambda}{2}\sum_{j=1}^{n_u}\sum_{k=1}^{n}(\theta_k^{(j)})^2 + \frac{\lambda}{2}\sum_{i=1}^{n_m}\sum_{k=1}^{n}(x_k^{(i)})^2$.
* The limitations of collaborative filtering are:
  * It is difficult to recommend new items.
  * It is difficult to recommend items to new users.
  * It is difficult to explain why a particular recommendation was made.

#### Content-based Filtering
* __Content-based filtering__ is a type of recommender system that makes predictions based on the similarity between the past behavior of the user and a particular item.
* There are 2 main steps in building a model for content-based filtering, namely retrieving and ranking.
  * __Retrieving__ is the process of finding items that are similar to the item that the user is interested in.
  * __Ranking__ is the process of ranking the retrieved items based on how similar they are to the item that the user is interested in.
* It is important to be ethical when building recommender systems, such as prioritizing value added to users rather than exploitation for profit.
  * This is particularly visible in the advertising industry, where it is difficult to distinguish between ads that are relevant to the user and ads that are simply trying to exploit the user.
  * One way to increase the likelihood of being ethical is to be transparent to users about how the system works.

#### Principal Component Analysis
* __Principal component analysis__ is a dimensionality reduction algorithm that can be used to reduce the dimensionality of a dataset.
  * The algorithm works by:
    1. Normalize the data.
    2. Compute the covariance matrix of the data.
    3. Compute the eigenvectors and eigenvalues of the covariance matrix.
    4. Sort the eigenvectors by decreasing eigenvalues and choose the k eigenvectors with the largest eigenvalues to form a matrix U of size n x k.
    5. Use this matrix to transform the data into the k-dimensional space.

### <ul>Week 3: Reinforcement Learning</ul>
#### Reinforcement Learning Introduction
* __Reinforcement learning__ is a type of machine learning algorithm where the goal is to learn how to maximize some reward.
* It has been successfully used in:
  * Robotics.
  * Game playing.
  * Finance.
  * and many more areas.
* __Policies__ in reinforcement learning are the rules that the agent follows to choose an action.
* __Markov Decision Process__ is a model that can be used to represent a reinforcement learning problem.
  * The model consists of:
    * A set of states.
    * A set of actions.
    * A reward function.
    * A transition model.
  * The goal of reinforcement learning is to find a policy that maximizes the expected reward.

#### State-action value function
* __State-action value function__ is a function that maps from a state and an action to a real number.
  * The value of a state-action pair is the expected reward that the agent will receive if it is in that state and takes that action.
  * The value of a state is the maximum value of the state-action pairs for that state.
  * Also known as the __Q-function__.
* __The Bellman equation__ is an equation that can be used to compute the value of a state-action pair.
  * The equation is $Q(s,a) = r(s,a) + \gamma\sum_{s'}P(s'|s,a)max_{a'}Q(s',a')$.
    * r(s,a) is the reward for taking action a in state s.
    * $\gamma$ is the discount factor.
    * P(s'|s,a) is the probability of transitioning from state s to state s' when taking action a.
    * max is the maximum operator.

#### Continuous State Spaces
* __Continuous state spaces__ are state spaces that are not discrete.
* The algorithm for computing the value of a state-action pair in a continuous state space is:
  1. Initialize Q arbitrarily.
  2. For each episode:
    1. Initialize s.
    2. For each step of the episode:
      1. Choose a from s using policy derived from Q (e.g. $\epsilon$-greedy).
      2. Take action a, observe r, s'.
      3. Q(s,a) = Q(s,a) + $\alpha$(r + $\gamma$max$_{a'}$Q(s',a') - Q(s,a)).
      4. s = s'.
* An __epsilon-greedy policy__ is a policy that chooses a random action with probability $\epsilon$ and chooses the action with the highest value with probability 1 - $\epsilon$.


# References
* [Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction)
