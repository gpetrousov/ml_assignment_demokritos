# 2.0 Course - Machine Learning assignment

Topics are based on the learning types.

![[Pasted image 20251111132201.png]]


# Potential topics - Supervised

#### 1. Binary Classification : Predicting Overtaking Difficulty in F1 🚥

- Use data from ErgastAPI (fastf1) - pulls data from the official F1 systems
- Inject weather data?
- Correlate with videos which show overtake or not.

#### 2. Regression Project: Predicting Pit Stop Duration ⏱️

- Goal: Predict the total duration of a pit stop (in seconds) based on pre-stop conditions.
- Correlate with videos which pit stops.

#### 3. Clustering Project: Unsupervised Strategy Analysis 🧩

- Use KNN to identify distinct strategies used by the teams for a race.


#### 4. Music genre classification + PCA 🎵

- Algos: Random Forest, SVM
- Connects well with PCA to reduce features
- Dataset: https://www.kaggle.com/datasets/carlthome/gtzan-genre-collection


# Potential topics - UNsupervised

Use Q-learning and SARSA (algorithms) in the following projects:

#### 1. Classic cart pole problem _|_

- add random collisions to the cart
- inject wind conditions to shift the pole
- inject gravity variation

https://gymnasium.farama.org/environments/classic_control/cart_pole/


#### 3. Humanoid walking 🤖

Teach a humanoid to walk to a point in a 2D environment.
- add a marker where the humanoid must walk to
- inject gravity differentiation
- add obstacles


#### 4. Catch the ball with... a twist 🌀

2D environment in which an agent is trying to learn to catch a ball.
- add physics
- inject multiple agents
- add human player