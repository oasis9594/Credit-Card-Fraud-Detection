import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import pandas as pd
from pandas_ml import ConfusionMatrix
import datetime
import pandas_ml as pdml
import imblearn
# Import PySwarms
import pyswarms as ps

# Some more magic so that the notebook will reload external python modules;
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
%load_ext autoreload
%autoreload 2

df = pd.read_csv('creditcard.csv', low_memory=False)
df = df.drop(['Time','Amount'],axis=1)
X = df.iloc[:,:-1]
y = df['Class']

# number_records_fraud = len(df[df.Class == 1])
# fraud_indices = np.array(df[df.Class == 1].index)

# # Picking the indices of the normal classes
# normal_indices = df[df.Class == 0].index

# # Out of the indices we picked, randomly select "x" number (number_records_fraud)
# random_normal_indices = np.random.choice(normal_indices, 3*number_records_fraud, replace = False)
# random_normal_indices = np.array(random_normal_indices)

# # Appending the 2 indices
# under_sample_indices = np.concatenate([fraud_indices,random_normal_indices])

# # Under sample dataset
# under_sample_data = df.iloc[under_sample_indices,:]

# X = under_sample_data.ix[:, under_sample_data.columns != 'Class']
# y = under_sample_data.ix[:, under_sample_data.columns == 'Class']

from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

data = scale(X)
pca = PCA(n_components=10)
X = pca.fit_transform(data)
print(X)
print(y)
print(type(X))
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
df2 = pdml.ModelFrame(X_train, target=y_train)
sampler = df2.imbalance.over_sampling.SMOTE()
oversampled = df2.fit_sample(sampler)
X, y = oversampled.iloc[:,1:11], oversampled['Class']

print(X)
print(type(X))
X=X.as_matrix()
y=y.as_matrix()
print(X)
print(type(X))

def forward_prop(params):
    """Forward propagation as objective function

    This computes for the forward propagation of the neural network, as
    well as the loss. It receives a set of parameters that must be
    rolled-back into the corresponding weights and biases.

    Inputs
    ------
    params: np.ndarray
        The dimensions should include an unrolled version of the
        weights and biases.

    Returns
    -------
    float
        The computed negative log-likelihood loss given the parameters
    """
    # Neural network architecture
    n_inputs = 10
    n_hidden = 20
    n_classes = 2

    # Roll-back the weights and biases
    W1 = params[0:200].reshape((n_inputs,n_hidden))
    b1 = params[200:220].reshape((n_hidden,))
    W2 = params[220:260].reshape((n_hidden,n_classes))
    b2 = params[260:262].reshape((n_classes,))

    #print(W1)
    #print(W2)

    # Perform forward propagation
    z1 = X.dot(W1) + b1  # Pre-activation in Layer 1
    a1 = np.tanh(z1)     # Activation in Layer 1
    z2 = a1.dot(W2) + b2 # Pre-activation in Layer 2
    logits = z2          # Logits for Layer 2

    # Compute for the softmax of the logits
    exp_scores = np.exp(logits)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    # Compute for the negative log likelihood
    N = len(X) # Number of samples
    corect_logprobs = -np.log(probs[range(N), y])
    loss = np.sum(corect_logprobs) / N
    return loss

def f(x):
    """Higher-level method to do forward_prop in the
    whole swarm.

    Inputs
    ------
    x: numpy.ndarray of shape (n_particles, dimensions)
        The swarm that will perform the search

    Returns
    -------
    numpy.ndarray of shape (n_particles, )
        The computed loss for each particle
    """
    n_particles = x.shape[0]
    j = [forward_prop(x[i]) for i in range(n_particles)]
    print(datetime.datetime.now().time())
    return np.array(j)

# Initialize swarm
options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}

# Call instance of PSO
dimensions = 262
optimizer = ps.single.GlobalBestPSO(n_particles=50, dimensions=dimensions, options=options)

# Perform optimization
cost, pos = optimizer.optimize(f, print_step=1, iters=150, verbose=3)

#Pass X, pos to check for training set and X_test, pos for testing set
def predict(X, pos):
    """
    Use the trained weights to perform class predictions.

    Inputs
    ------
    X: numpy.ndarray
        Input Iris dataset
    pos: numpy.ndarray
        Position matrix found by the swarm. Will be rolled
        into weights and biases.
    """
    # Neural network architecture
    n_inputs = 10
    n_hidden = 20
    n_classes = 2

    # Roll-back the weights and biases
    W1 = pos[0:200].reshape((n_inputs,n_hidden))
    b1 = pos[200:220].reshape((n_hidden,))
    W2 = pos[220:260].reshape((n_hidden,n_classes))
    b2 = pos[260:262].reshape((n_classes,))

    # Perform forward propagation
    z1 = X.dot(W1) + b1  # Pre-activation in Layer 1
    a1 = np.tanh(z1)     # Activation in Layer 1
    z2 = a1.dot(W2) + b2 # Pre-activation in Layer 2
    logits = z2          # Logits for Layer 2

    y_pred = np.argmax(logits, axis=1)
    return y_pred
