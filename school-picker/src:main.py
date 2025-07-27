from school_picker import SchoolPicker
from dqn_student import DQNStudent

# Initialize environment (school picker) and agent (instance of student)
env = SchoolPicker(drop_columns=True)
agent = DQNStudent(state_size=env.num_features, action_size=len(env.remaining))

# Initialize new instance
state = env.restart()

# Begin infinite loop
while True:
    # Display current environment
    env.display()
    
    # Choose action based on current state
    action = agent.choose_action(state)
    # Applies action to environment and collects results
    next_state, reward, done = env.make_choice(action)
    
    # Takes swipe decision
    feedback = input("Yes or no. ").strip().lower()
    # If user says yes
    if feedback == 'yes':
        # Apply positive reward
        reward = 1
        # Tracks if feedback was received
        got_feedback = True
    # If user says no
    elif feedback == 'no':
        # Apply negative reward
        reward = -1
        # Tracks if feedback was received
        got_feedback = False
    # If any other input was received
    else:
        # Take it as a no
        print("Invalid input, received as a no.")
        # Apply negative reward
        reward = -1
        # Tracks if feedback was received
        got_feedback = False

    # Stores experience in memory
    agent.remember(state, action, reward, next_state, done)
    # Trains agent using batch of experiences
    agent.train(50)
    # Adjusts epsilon
    agent.adjust_epsilon(got_feedback)
    # Updates state
    state = next_state