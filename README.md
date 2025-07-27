# School picker with Deep Q-Learning

A reinforcement learning project that recommends colleges to students based on real-world institutional data and user preferences. Built using Python and TensorFlow, this interactive program simulates a "swipe-style" decision process and learns which schools users prefer based on their own selection.

## Motivation

The tool provides an in-depth application for prospective college students to get a better idea of schools that they could be interested in. Reinforcement learning (specifically Deep Q-Learning) was implemented in order to optimize an interactive platform that students can look at possible universities. The student views a single school at a time, along with the features associated with it. The model slowly adapts to the user’s preference and provides more schools as the swipe count increases.

# How it works
- Loads real college data from the Department of Education. 
- The user interacts by saying “yes” or “no” to given school suggestion.
- Deep Q-Network learns to predict schools to match the user’s preferences based on features such as tuition, size, and more.
- The model adjusts the Q-value over time to optimize the provided school to the user.

## Features
— Deep Q-Learning agent using TensorFlow
— Real-time user feedback loop
— Dynamic state updates and memory replay
— Preprocessing of real-world data
— Modular code structure (OOP)

## Technologies Used
— Python 3
— TensorFlow / Keras
— NumPy / Pandas
