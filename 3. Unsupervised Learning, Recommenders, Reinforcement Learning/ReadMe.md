# Part 3: [Unsupervised Learning, Recommenders, Reinforcement Learning](./3.%20Unsupervised%20Learning%2C%20Recommenders%2C%20Reinforcement%20Learning/)
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
