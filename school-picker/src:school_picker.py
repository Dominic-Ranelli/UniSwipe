import numpy as np
import pandas as pd
class SchoolPicker:

    """

A simulated environment for selecting schools based on feature vectors and user feedback.

This environment utilizes a reinforcement learning loop where an agent interacts with college data and gets rewards based on their user preference.

Attributes:
    data (pd.DataFrame): Processed numerical dataset of colleges.
    names (List[str]): List of school names.
    features (np.ndarray): Matrix of feature vectors for each school.
    name_to_index (Dict[str, int]): Maps school names to row indices.
    num_features (int): Number of features per school.
    num_schools (int): Total number of schools after preprocessing. 

""" 

    def __init__(self, drop_columns=False):
        """
        Initializes the environment, loads and preprocesses the dataset.

        Args:
            drop_columns (bool): If True, drops columns with >50% missing data.
        """



        self.data = pd.read_csv('/Users/dominicranelli/Downloads/Most-Recent-Cohorts-Institution.csv', index_col='INSTNM')
        
        if drop_columns:
            self._drop_columns()
        
        self._handle_non_numeric_data()

        self.data = self.data.apply(pd.to_numeric, errors='coerce')  # Convert to numeric, set errors='coerce' to turn non-numeric into NaN

        self.data = self.data.dropna()

        self.names = self.data.index.tolist()
        self.features = self.data.values
        self.name_to_index = {name: i for i, name in enumerate(self.names)}
        self.num_features = self.features.shape[1]
        self.num_schools = self.features.shape[0]
        self.restart()

    def _drop_columns(self):
        # Drops columns with more than 50% of data missing.
        missing_perc = self.data.isnull().mean() * 100
        self.data = self.data.loc[:, missing_perc < 50]
        self.features = self.data.values


    def _handle_non_numeric_data(self):
        # Convert categorical columns to numeric using one-hot encoding or label encoding.
        # Called automatically during initialization.
        categorical_columns = self.data.select_dtypes(include=['object']).columns
        if not categorical_columns.empty:
            self.data = pd.get_dummies(self.data, columns=categorical_columns)

    def restart(self):
        # Resets environment state and intializes a new episode.
        # np.ndarray: Returns the initial state vector.
        self.remaining = set(self.names)
        self.rejected = set()
        self.current_state = np.zeros(self.num_features)
        self.current_school = self._get_best_school()
        if self.current_school is not None:
            self.current_state = self.features[self.name_to_index[self.current_school]]
        return self.current_state
    
    def _get_best_school(self):
        # Determines the best school to recommend based on the current state.
        # str or None: School name with highest dot-product score, or None if done.
        valid_schools = self.remaining - self.rejected
        if not valid_schools:
            return None
        scores = [np.dot(self.features[self.name_to_index[name]], self.current_state) for name in valid_schools]
        best_school = max(valid_schools, key=lambda name: np.dot(self.features[self.name_to_index[name]], self.current_state))
        return best_school
    
    def make_choice(self, school_name):
        # Applies user's decision and returns  resulting environment state.
        # Argument: school_name (str): Name of the school being evaluated.
        # Returns: tuple (next_state, reward, done)
        if school_name not in self.remaining:
            return self.current_state, 0, True

        index = self.name_to_index[school_name]
        reward = np.dot(self.features[index], self.current_state)
        self.remaining.remove(school_name)
        if reward < 0:
            self.rejected.add(school_name)
        done = (len(self.remaining) == 0)
        if not done:
            self.current_school = self._get_best_school()
            if self.current_school is not None:
                self.current_state = self.features[self.name_to_index[self.current_school]]
        else:
            self.current_state = np.zeros(self.num_features)
        return self.current_state, reward, done
    
    def display(self):
        # Prints name of the current recommended school.
        if self.current_school is not None:
            print(f"Current pick: {self.current_school}")
        else:
            print("Schools remaining: None.")