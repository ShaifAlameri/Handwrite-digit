
import tensorflow as tf

# Load the MNIST dataset
mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize the pixel values of the images
X_train = tf.keras.utils.normalize(X_train, axis=1)
X_test = tf.keras.utils.normalize(X_test, axis=1)

# Create a sequential model
model = tf.keras.models.Sequential()

# Add layers to the model
# Flatten layer: Flattens the input images from 2D to 1D
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))

# Dense layer: Fully connected layer with 128 units and ReLU activation function
model.add(tf.keras.layers.Dense(128, activation='relu'))

# Dense layer: Another fully connected layer with 128 units and ReLU activation function
model.add(tf.keras.layers.Dense(128, activation='relu'))

# Dense layer: Final output layer with 10 units (corresponding to the 10 digit classes) and softmax activation function
model.add(tf.keras.layers.Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=3)

# Evaluate the model on the test data
loss, accuracy = model.evaluate(X_test, y_test)

# Save the trained model
model.save('handwrittendigit.model')