import numpy as np
import random
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


class DQNStudent:

    """
        Deep Q-Network student agent, learns to select schools based on user preferences (swipes).

        Attributes:
            state_size (int): Length of state vector.
            action_size (int): Number of possible actions (schools).
            memory (deque): Experience replay buffer.
            gamma (float): Discount factor.
            alpha (float): Learning rate.
            epsilon (float): Exploration probability.
            epsilon_min (float): Minimum exploration.
            epsilon_decay (float): Decay rate for epsilon.
            model (tf.keras.Model): Neural network model for Q-learning.
    """

    # Building of NN
    def __init__(self, state_size, action_size):
        """
        Initializes DQN agent with given state and action sizes.

        Args:
            state_size (int): Dimensionality of input state.
            action_size (int): Number of actions.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2500)
        self.gamma = 0.95
        self.alpha = 0.001
        self.epsilon = 1.0
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.99
        self.model = self._build_model()

    def _build_model(self):
        # Builds and compiles neural network for Q-value approximation.
        model = Sequential()
        model.add(Dense(24, activation='relu', input_shape=(self.state_size,)))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.alpha))
        return model
    
    def remember(self, state, action, reward, next_state, done):
        # Stores single experience in memory.
        # Args:
            # state (np.ndarray): Current state.
            # action (str): Action taken (school name).
            # reward (float): Reward received.
            # next_state (np.ndarray): Next state.
            # done (bool): Whether the episode has ended.
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state):
        # Chooses next action based on policy.
        # Arguments: 
            # state (np.ndarray): Current env state.
            # env (SchoolPicker): Environment instance to extract remaining schools.
        # Returns: str: Selected school name.
        if np.random.rand() <= self.epsilon:
            return env._get_best_school()
        q_values = self.model.predict(np.expand_dims(state, axis=0))
        best_action_index = np.argmax(q_values[0])
        return list(env.remaining)[best_action_index]
    
    def train(self, batch_size):
        # Trains model using experience replay.
        # Args:
            # batch_size (int): Number of experiences sampled.
            # env (SchoolPicker): Env used to call action indexes.
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.gamma * np.amax(self.model.predict(np.expand_dims(next_state, axis=0))[0])
            target_f = self.model.predict(np.expand_dims(state, axis=0))
            action_index = list(env.remaining).index(action)
            target_f[0][action_index] = target
            self.model.fit(np.expand_dims(state, axis=0), target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


    def adjust_epsilon(self, got_feedback):
        # Adjusts exploration rate based on feedback received.
        # Args: got_feedback (bool): True if user sent feedback.
        if got_feedback:
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
        else:
            if self.epsilon < 1.0:
                self.epsilon += (1 - self.epsilon) * 0.1
