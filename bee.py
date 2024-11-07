 xaiver and kaiming 

import tensorflow as tf

from tensorflow.keras import layers, models, initializers

from tensorflow.keras.datasets import cifar10

from tensorflow.keras.utils import to_categorical

import matplotlib.pyplot as plt

# Step 1: Load CIFAR-10 dataset

(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

train_images, test_images = train_images / 255.0, test_images / 255.0

train_labels, test_labels = to_categorical(train_labels), to_categorical(test_labels)

# Step 2: Preprocess the Data

# ...

# Step 3: Define the Neural Network Architecture

def create_model(initializer, dropout_rate=0.0, l2_regularizer=None):

    model = models.Sequential()

    model.add(layers.Flatten(input_shape=(32, 32, 3)))

    model.add(layers.Dense(512, kernel_initializer=initializer, kernel_regularizer=l2_regularizer, activation='relu'))

    model.add(layers.Dense(256, kernel_initializer=initializer, kernel_regularizer=l2_regularizer, activation='relu'))

    model.add(layers.Dense(128, kernel_initializer=initializer, kernel_regularizer=l2_regularizer, activation='relu'))

    model.add(layers.Dense(64, kernel_initializer=initializer, kernel_regularizer=l2_regularizer, activation='relu'))  # Additional dense layer

    model.add(layers.Dense(32, kernel_initializer=initializer, kernel_regularizer=l2_regularizer, activation='relu'))  # Additional dense layer

    model.add(layers.Dense(10, activation='softmax'))

    return model

# Step 4: Choose Weight Initialization Techniques

xavier_initializer = initializers.glorot_normal()

kaiming_initializer = initializers.he_normal()

 

# Step 5: Compile the Model

# ...

# Step 6: Train the Model with Different Configurations

xavier_model = create_model(xavier_initializer, dropout_rate=0.3, l2_regularizer=tf.keras.regularizers.l2(0.001))

kaiming_model = create_model(kaiming_initializer, dropout_rate=0.3, l2_regularizer=tf.keras.regularizers.l2(0.001))

xavier_model.compile(optimizer='adam',

                     loss='categorical_crossentropy',

                     metrics=['accuracy'])

kaiming_model.compile(optimizer='adam',

                      loss='categorical_crossentropy',

                      metrics=['accuracy'])

xavier_history = xavier_model.fit(train_images, train_labels, epochs=20, validation_data=(test_images, test_labels))

kaiming_history = kaiming_model.fit(train_images, train_labels, epochs=20, validation_data=(test_images, test_labels))

# Step 7: Evaluate and Visualize Performance

# ...

# Step 8: Display Output

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)

plt.plot(xavier_history.history['accuracy'], label='Xavier (train)')

plt.plot(xavier_history.history['val_accuracy'], label='Xavier (val)')

plt.title('Xavier Initialization')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend()

plt.subplot(1, 2, 2)

plt.plot(kaiming_history.history['accuracy'], label='Kaiming (train)')

plt.plot(kaiming_history.history['val_accuracy'], label='Kaiming (val)')

plt.title('Kaiming Initialization')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend()

plt.tight_layout()

plt.show()






MNIST DIGIT CLA PROGRAM:

import tensorflow as tf

from tensorflow.keras.datasets import mnist

from tensorflow.keras.utils import to_categorical

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images, test_images = train_images / 255.0, test_images / 255.0

train_labels, test_labels = to_categorical(train_labels), to_categorical(test_labels)

model = tf.keras.Sequential([

 tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),

 tf.keras.layers.MaxPooling2D(),

 tf.keras.layers.Conv2D(64, 3, activation='relu'),

 tf.keras.layers.MaxPooling2D(),

 tf.keras.layers.Conv2D(64, 3, activation='relu'),

 tf.keras.layers.Flatten(),

 tf.keras.layers.Dense(64, activation='relu'),

 tf.keras.layers.Dense(10, activation='softmax')

])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_split=0.2)

_, test_acc = model.evaluate(test_images, test_labels)

print(f"\nTest Accuracy : {round(test_acc * 100, 4)}%")


VGGNET-19 PROGRAM:

import numpy as np

import tensorflow as tf

mnist=np.load("/content/mnist.npz")

x_train=mnist["x_train"]

y_train=mnist["y_train"]

x_test=mnist["x_test"]

y_test=mnist["y_test"]

y_test = tf.keras.utils.to_categorical(y_test)

y_train = tf.keras.utils.to_categorical(y_train)

npad = ((0, 0), (10, 10), (10, 10))

x_train=np.pad(x_train, npad, 'constant', constant_values=(255))

x_train=np.array([np.stack((img,img,img), axis=-1) for img in x_train])

from tensorflow.keras import models,layers

from tensorflow.keras.applications import VGG16

vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(48, 48, 3))

 

# Add custom classification layers

model = models.Sequential()

model.add(vgg_model)

model.add(layers.Flatten())

model.add(layers.Dense(128, activation='relu'))

model.add(layers.Dense(10, activation='softmax'))


model.compile(optimizer='adam',

              loss='categorical_crossentropy',

              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)



COMPARITIVE OF LSTM,GRU AND RNN PROGRAM:

import numpy as np

import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import SimpleRNN, LSTM, GRU, Dense

from sklearn.metrics import mean_squared_error

 

# Generate sample data

np.random.seed(42)

n_samples = 1000

time_steps = 20

 

# Generate a simple sinusoidal time series

t = np.linspace(0, 10, n_samples, endpoint=False)

data = np.sin(t) + 0.1 * np.random.randn(n_samples)

 

# Create sequences of data with corresponding targets

sequences = []

targets = []

for i in range(n_samples - time_steps):

    seq = data[i : i + time_steps]

    target = data[i + time_steps]

    sequences.append(seq)

    targets.append(target)

 

# Convert to numpy arrays

sequences = np.array(sequences)

targets = np.array(targets)

 

# Reshape the input data for RNNs

sequences = sequences.reshape(-1, time_steps, 1)

 

# Split the data into training and testing sets

split = int(0.8 * n_samples)

X_train, X_test = sequences[:split], sequences[split:]

y_train, y_test = targets[:split], targets[split:]

 

# Function to build and train the model

def build_and_train_model(model_type):

    model = Sequential()

    if model_type == "SimpleRNN":

        model.add(SimpleRNN(50, activation="relu", input_shape=(time_steps, 1)))

    elif model_type == "LSTM":

        model.add(LSTM(50, activation="relu", input_shape=(time_steps, 1)))

    elif model_type == "GRU":

        model.add(GRU(50, activation="relu", input_shape=(time_steps, 1)))

    else:

        raise ValueError("Invalid model type")

 

    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")

 

    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=0)

   

    return model, history

 

# Train models

rnn_model, rnn_history = build_and_train_model("SimpleRNN")

lstm_model, lstm_history = build_and_train_model("LSTM")

gru_model, gru_history = build_and_train_model("GRU")

 

# Evaluate models on the test set

rnn_pred = rnn_model.predict(X_test)

lstm_pred = lstm_model.predict(X_test)

gru_pred = gru_model.predict(X_test)

 

# Calculate Mean Squared Error

rnn_mse = mean_squared_error(y_test, rnn_pred)

lstm_mse = mean_squared_error(y_test, lstm_pred)

gru_mse = mean_squared_error(y_test, gru_pred)

 

# Plot performance comparison

plt.plot(rnn_history.history["loss"], label="SimpleRNN Training Loss")

plt.plot(lstm_history.history["loss"], label="LSTM Training Loss")

plt.plot(gru_history.history["loss"], label="GRU Training Loss")

plt.legend()

plt.title("Training Loss Comparison")

plt.xlabel("Epochs")

plt.ylabel("Mean Squared Error")

plt.show()

 

print(f"Mean Squared Error on Test Set:")

print(f"SimpleRNN: {rnn_mse}")

print(f"LSTM: {lstm_mse}")

print(f"GRU: {gru_mse}")

 

TIME SERIES FORCASTING FOR NIFTY 50 PROGRAM:

import pandas as pd

import numpy as np

import yfinance as yf

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error

 

# Download historical NIFTY 50 data using yfinance

nifty_data =yf.download('^NSEI', start='2010-01-01', end='2022-01-01')

 

# Extract relevant columns

nifty_data = nifty_data[['Adj Close']]

 

# Reset the index to make Date a column

nifty_data = nifty_data.reset_index()

 

# Feature engineering: Adding a column for days since the start

nifty_data['Days'] = (nifty_data['Date'] - nifty_data['Date'].min()).dt.days

 

# Split the data into training and testing sets

train, test = train_test_split(nifty_data, test_size=0.2, shuffle=False)

 

# Define the features (X) and target variable (y) for training

X_train = train[['Days']]

y_train = train['Adj Close']

 

# Define the features (X) and target variable (y) for testing

X_test = test[['Days']]

y_test = test['Adj Close']

 

# Initialize and fit the linear regression model

model = LinearRegression()

model.fit(X_train, y_train)

 

# Make predictions on the test set

y_pred = model.predict(X_test)

 

# Evaluate the model using Mean Squared Error (MSE)

mse = mean_squared_error(y_test, y_pred)

print(f'Mean Squared Error: {mse}')

 

# Plot the actual vs. predicted values

plt.figure(figsize=(10, 6))

plt.plot(test['Date'], y_test, label='Actual')

plt.plot(test['Date'], y_pred, label='Predicted', linestyle='dashed')

plt.xlabel('Date')

plt.ylabel('Adj Close')

plt.title('NIFTY 50 Time Series Forecasting - Linear Regression')

plt.legend()

plt.show()

 

CIFAR 10


import tensorflow as tf

from tensorflow.keras import layers, models

from tensorflow.keras.datasets import cifar10

from tensorflow.keras.utils import to_categorical

import numpy as np

import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0

y_train, y_test = to_categorical(y_train), to_categorical(y_test)

def create_model(hidden_units=None, activation=None):

    model = models.Sequential([

        layers.Flatten(input_shape=(32, 32, 3)),

        layers.Dense(hidden_units[0], activation=activation),

        layers.Dense(hidden_units[1], activation=activation),

        layers.Dense(hidden_units[2], activation=activation),

        layers.Dense(10, activation='softmax')

    ])

    return model

hidden_units = [512, 256, 128]

activation = 'relu'

results_dict = {}

counter = 1

model = create_model(hidden_units=hidden_units, activation=activation)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test))

_, test_acc = model.evaluate(x_test, y_test)

model_info = {

    "Hidden units": hidden_units,

    "Activation": activation,

    "Test accuracy": round(test_acc * 100, 4)

}

results_dict[counter] = model_info 

counter += 1

for key, value in results_dict.items():

    print(f"Run {key}:")

    for info_key, info_value in value.items():

        print(f"{info_key}: {info_value}")

    print("- -" * 15)

print("\n")

max_accuracy_run = max(results_dict, key=lambda k: results_dict[k]["Test accuracy"])

max_accuracy_info = results_dict[max_accuracy_run]

print("Run with the highest test accuracy:")

print(f"Run {max_accuracy_run}:")

for info_key, info_value in max_accuracy_info.items():

    print(f"{info_key}: {info_value}")

num_images = 3

sample_images = x_train[:num_images]

predictions = model.predict(sample_images)

def plot_probability_meter(predictions, image):

    class_labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    fig, axs = plt.subplots(1, 2, figsize=(10, 2))

    axs[0].imshow(image)

    axs[0].axis('off')

    axs[1].barh(class_labels, predictions[0], color='blue')

    axs[1].set_xlim([0, 1])

    plt.tight_layout()

    plt.show()

for i in range(num_images):

    plot_probability_meter(predictions[i:i+1], sample_images[i])
