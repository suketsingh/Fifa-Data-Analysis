

#####
#####  Importing the needed packages
#####

import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, Dropout, LSTM
from keras.datasets import imdb
from keras import backend as K
import pyswarms as ps
from keras.datasets import imdb
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix
from sklearn import preprocessing, datasets, linear_model
from sklearn.model_selection import train_test_split

#####
#####  Loading the dataset and bifercating it into training and testing data for IMDB dataset
#####

# maximum vocabulary
nb_words = 50000
# cut texts after this number of words
maxlen = 100

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=nb_words)
x = x_train
y = y_train

#####
#####  Padding input sequences
#####

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

#####
#####  Shaping the training and data input
#####

print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

#####
#####  Addign the model parameters for the Neural Network
#####  Making the layers of the Neural Network
#####

model = Sequential()
model.add(Dense(20, input_dim=100, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))

#####
#####  Forward propagation is used as the objective function
#####  This function is used to compute the Forward Propagation of the Neural Network
#####  It also computes the loss for the function
#####  A set of parameters are rolled back to the corresponding weights and biases
#####  Input to the function:
#####  Params: np.ndarray
#####  The dimensions should include an unrolled version of the weights and biases.
#####  Returns from the function:
#####  The computed negative log-likelihood loss given the parameters
#####

def forward_prop(params):
    # Neural network architecture defining
    n_inputs = 100
    n_hidden = 20
    n_classes = 1

    # Rolling back the weights and biases
    W1 = params[0:2000].reshape((n_inputs,n_hidden))
    b1 = params[2000:2020].reshape((n_hidden,))
    W2 = params[2020:2040].reshape((n_hidden,n_classes))
    b2 = params[2040:2041].reshape((n_classes,))

    # Perform forward propagation
    z1 = x_train.dot(W1) + b1  # Pre-activation in Layer 1
    a1 = np.tanh(z1)     # Activation in Layer 1
    z2 = a1.dot(W2) + b2 # Pre-activation in Layer 2
    logits = z2          # Logits for Layer 2

    # Compute for the softmax of the logits
    exp_scores = np.exp(logits)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    # Compute for the negative log likelihood
    N = 25000 # Number of samples
    corect_logprobs = -np.log(probs[range(N)])
    loss = np.sum(corect_logprobs) / N

    return loss

#####
#####  This is a higher-level method to compute forward_prop() to the whole swarm
#####  Inputs:
#####  params: np.ndarray
#####  The dimensions should include an unrolled version of the weights and biases.
#####  Returns:
#####  The computed negative log-likelihood loss given the parameters
#####

def f(x):
    n_particles = x.shape[0]
    j = [forward_prop(x[i]) for i in range(n_particles)]
    return np.array(j)

#####
#####  Calling global-best PSO and running the optimizer
#####

# Initialize swarm
options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}

# Call instance of PSO 500 input -> 20 hidden layer -> 1 output
dimensions = (100 * 20) + (20 * 1) + 20 + 1
optimizer = ps.single.GlobalBestPSO(n_particles=100, dimensions=dimensions, options=options)

# Perform optimization
cost, pos = optimizer.optimize(f, iters=25)

#####
#####  Checking the accuracy once again to create a set of predictions
#####  The best position vector is found by the swarm which tend to be the weight and bias parameter of the network itself
#####

# Neural network architecture
n_inputs = 100
n_hidden = 20
n_classes = 1

W1 = pos[0:2000].reshape((n_inputs,n_hidden))
b1 = pos[2000:2020].reshape((n_hidden,))
W2 = pos[2020:2040].reshape((n_hidden,n_classes))
b2 = pos[2040:2041].reshape((n_classes,))

model.layers[0].set_weights([W1,b1])
model.layers[1].set_weights([W2,b2])

predictions = model.predict(x_test, batch_size=20, verbose=0)
predictions_classes = model.predict_classes(x_test, batch_size=20, verbose=0)

print('Mean Squared error for Predictions:')
print(mean_squared_error(y_test, predictions_classes))
print('Cost:')
print(cost)

#####
#####  Printing the Confusion Matrix
#####

print('Confusion Matrix:')
roundedy = [round(x[0]) for x in predictions_classes]
print(confusion_matrix(y, predictions_classes))


