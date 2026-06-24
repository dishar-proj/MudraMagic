import numpy as np
import tensorflow as tf

# This script is used to read the CSV file to train a dense neural network.
# It compares results to testing data and the model is saved as a .keras file.

print("Loading datasets...") # terminal output for status updates

# Get the training data from the CSV files
try:
    # Load Training Data, ignore the first row header 
    train_data = np.loadtxt('train_landmarks.csv', delimiter=',', skiprows=1)
    y_train = train_data[:, 0]
    X_train = train_data[:, 1:]

    # Load Testing Data
    test_data = np.loadtxt('test_landmarks.csv', delimiter=',', skiprows=1)
    y_test = test_data[:, 0]
    X_test = test_data[:, 1:]
except FileNotFoundError: # throws an error if csv files are not found
    print("Error: Could not find the CSV files. Run extract_landmarks.py first!")
    exit()

# Shuffle ONLY the training data to reduce bias, this is so not 
# all the pataka files are fed first into the program instead it is randomized
train_indices = np.arange(X_train.shape[0])
np.random.shuffle(train_indices)
X_train = X_train[train_indices]
y_train = y_train[train_indices]

print(f"Training on {len(X_train)} skeletons, validating on {len(X_test)} skeletons.")

# Define the dense neural network
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(63,)), # takes 63 numbers = 21 joins * 3 coordinates (x, y,z)      
    tf.keras.layers.Dense(128, activation='relu'),   # 128 neurons to find patterns       
    tf.keras.layers.Dropout(0.2), # to avoid overfitting issues encountered turn of 20% of neurons during training                          
    tf.keras.layers.Dense(64, activation='relu'), # 64 neurons to find more patterns          
    tf.keras.layers.Dense(10, activation='softmax')  # 10 possible mudras
])

# compile the ai
model.compile(optimizer='adam', # optimizier/learning agent
              loss='sparse_categorical_crossentropy', #calculate how wrong the network is
              metrics=['accuracy'])

print("\nTraining the model...")
# will view image deck 50 times
model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))

# Save the finished model to use for final application
model.save('landmark_mudra_model_10.keras')
print("\nSuccess! New brain saved as 'landmark_mudra_model_10.keras'")