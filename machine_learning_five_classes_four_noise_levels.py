# %%
import numpy as np

import jax
import jax.numpy as jnp
from jax import jit, grad, vmap

import pickle as pkl

import itertools

import os

from alive_progress import alive_bar
import gc

parallel_scan = jax.lax.associative_scan

import more_itertools as mit

# %% [markdown]
# ## Make a dataloader generator function

# %%
## Probably not that usefull
SNR = jnp.arange(0,20, 5)
frequencies = jnp.array([100, 200, 300, 400, 500])


def dataloader(batch_size):
    while True:
        for snr, freq in itertools.product(SNR, frequencies):
            with open(f"datasets/dataset_freq_{freq}_SNR_{snr}.pkl", "rb") as f:
                data = pkl.load(f)
            for i in range(0, len(data), batch_size):
                yield data[i:i+batch_size]

# %%
with open(f"datasets/datasets_5_tones_with_noise.pkl", "rb") as f:
    data = pkl.load(f)

# Randomize the dataset
perm = np.random.permutation(len(data))

data

# %%
shuffled_data = [data[i] for i in perm]

# %%
# Seperate into a train set and test set
del data
gc.collect()

# %%
shuffled_data[0]

# %% [markdown]
# ## Make a single LRU

# %%
def binary_operator_diag(element_i, element_j):
    a_i, bu_i = element_i
    a_j, bu_j = element_j

    return a_j * a_i, a_j * bu_i + bu_j


def init_lru_parameters(N, H, r_min = 0.0, r_max = 1, max_phase = 6.28):
    # N: state dimension, H: model dimension
    # Initialization of Lambda is complex valued distributed uniformly on ring
    # between r_min and r_max, with phase in [0, max_phase].

    u1 = np.random.uniform(size = (N,))
    u2 = np.random.uniform(size = (N,))

    nu_log = np.log(-0.5*np.log(u1*(r_max**2-r_min**2) + r_min**2))
    theta_log = np.log(max_phase*u2)

    # Glorot initialized Input/Output projection matrices
    B_re = np.random.normal(size=(N,H))/np.sqrt(2*H)
    B_im = np.random.normal(size=(N,H))/np.sqrt(2*H)
    C_re = np.random.normal(size=(H,N))/np.sqrt(N)
    C_im = np.random.normal(size=(H,N))/np.sqrt(N)
    D = np.random.normal(size=(H,))

    # Normalization
    diag_lambda = np.exp(-np.exp(nu_log) + 1j*np.exp(theta_log))
    gamma_log = np.log(np.sqrt(1-np.abs(diag_lambda)**2))

    return nu_log, theta_log, B_re, B_im, C_re, C_im, D, gamma_log


def forward_LRU(lru_parameters, input_sequence):
    # Unpack the LRU parameters
    nu_log, theta_log, B_re, B_im, C_re, C_im, D, gamma_log = lru_parameters

    # Initialize the hidden state
    Lambda = jnp.exp(-jnp.exp(nu_log) + 1j*jnp.exp(theta_log))
    B_norm = (B_re + 1j*B_im) * jnp.expand_dims(jnp.exp(gamma_log), axis=-1)
    C = C_re + 1j*C_im

    Lambda_elements = jnp.repeat(Lambda[None, ...], input_sequence.shape[0], axis=0)

    Bu_elements = jax.vmap(lambda u: B_norm @ u)(input_sequence)
    elements = (Lambda_elements, Bu_elements)
    _, inner_states = parallel_scan(binary_operator_diag, elements) # all x_k
    y = jax.vmap(lambda x, u: (C @ x).real + D * u)(inner_states, input_sequence)


    return y

def loss_fn(lru_parameters, input_sequence, target_sequence):
    y = forward_LRU(lru_parameters, input_sequence)
    return jnp.mean((y - target_sequence)**2)


def update(lru_parameters, input_sequence, target_sequence):
    return  grad(loss_fn)(lru_parameters, input_sequence, target_sequence)

# %%
# Create the MLP encoder

# %% [markdown]
# # Make the MLP model

# %%
def init_mlp_parameters(layers):
    # Initialize the MLP parameters
    parameters = []
    for i in range(len(layers)-1):
        W = np.random.normal(size=(layers[i], layers[i+1]))/np.sqrt(layers[i])
        b = np.zeros((layers[i+1],))
        parameters.append((W, b))

    return parameters

@jit
def forward_mlp(mlp_parameters, input, activation_function = jnp.tanh):
    # Forward pass of the MLP
    
    x = input

    for W, b in mlp_parameters:
        x = x @ W + b
        x = activation_function(x)

    return x

def forward_mlp_linear_layer(mlp_parameters, input, activation_function = jnp.tanh):
    
    x = input

    # Only apply the MLP up to the second last layer
    for W, b in mlp_parameters[:-1]:
        x = x @ W + b
        x = activation_function(x)

    # Apply the last layer without activation function
    W, b = mlp_parameters[-1]
    x = x @ W + b

    # Use the softmax function on the last layer
    x = jax.nn.softmax(x)

    return x


# %%
def max_pooling(sequence_to_pool):
    return jnp.max(sequence_to_pool, axis=0)

def mean_pooling(sequence_to_pool):
    return jnp.mean(sequence_to_pool, axis=0)

def sum_pooling(sequence_to_pool):
    return jnp.sum(sequence_to_pool, axis=0)

# %% [markdown]
# # Create the complete model

# %%
# Create the model
Linear_encoder_parameter  = init_mlp_parameters([1,5,10])
seconday_parameters = init_mlp_parameters([10,10,10])
LRU = init_lru_parameters(10, 10)
Linear_decoder_parameter = init_mlp_parameters([10,5,5])

model_parameters = [Linear_encoder_parameter, LRU, seconday_parameters, Linear_decoder_parameter]

def model_forward(input_sequence, parameters):
    Linear_encoder_parameter,  LRU, seconday_parameters, Linear_decoder_parameter = parameters

    x = forward_mlp(Linear_encoder_parameter, input_sequence)
    x = forward_LRU(LRU, x)
    x = forward_mlp(seconday_parameters, x)
    x = max_pooling(x)
    x = forward_mlp_linear_layer(Linear_decoder_parameter, x)

    return x

def one_hot(x, k, dtype=jnp.float32):
  """Create a one-hot encoding of x of size k."""
  return jnp.array(x[:, None] == jnp.arange(k), dtype)

def model_loss(input_sequence, target_sequence, parameters):
    y = model_forward(input_sequence, parameters)

    # Cross entropy loss
    return -jnp.mean(jnp.sum(target_sequence * jnp.log(y), axis=1))

@jit
def model_grad(input_sequence, target_sequence, parameters):
    return grad(model_loss, argnums=2)(input_sequence, target_sequence, parameters)

@jit
def parameter_update(parameters, gradients, learning_rate = 0.01):
    new_parameters = []
    im = []
    for parameter, gradient in zip(parameters[0], gradients[0]):
        im.append((parameter[0] - learning_rate * gradient[0], parameter[1] - learning_rate * gradient[1]))

    new_parameters.append(im)

    im = []
    for parameter, gradient in zip(parameters[1], gradients[1]):
        im.append(parameter - learning_rate * gradient)    

    new_parameters.append(im)

    im = []
    for parameter, gradient in zip(parameters[2], gradients[2]):
        im.append((parameter[0] - learning_rate * gradient[0], parameter[1] - learning_rate * gradient[1]))

    new_parameters.append(im)

    im = []
    for parameter, gradient in zip(parameters[3], gradients[3]):
        im.append((parameter[0] - learning_rate * gradient[0], parameter[1] - learning_rate * gradient[1]))

    new_parameters.append(im)

    return new_parameters


# Test batch model forward
batch_model_forward = vmap(model_forward, in_axes=(0, None))

@jit
def accuracy(input_sequences, target_sequences, parameters):
    y = batch_model_forward(input_sequences, parameters)
    return jnp.mean(jnp.argmax(y, axis=1) == jnp.argmax(target_sequences, axis=1))

# %%
# Seperate into a train set and test set
train_sequences = jnp.array([x[0] for x in shuffled_data[:int(0.8*len(shuffled_data))]]).reshape((int(0.8*len(shuffled_data)),20000,1))
test_sequences = jnp.array([x[0] for x in shuffled_data[int(0.8*len(shuffled_data)):]]).reshape((len(shuffled_data) - int(0.8*len(shuffled_data)),20000,1))

train_labels = one_hot(jnp.array([x[1] for x in shuffled_data[:int(0.8*len(shuffled_data))]])/100, 5)
test_labels = one_hot(jnp.array([x[1] for x in shuffled_data[int(0.8*len(shuffled_data)):]])/100, 5)

del shuffled_data
gc.collect()

# %%
epochs = 10
batchsize = 1

train_acc = []
test_acc = []



for i in range(epochs):
    print(f"Epoch {i}")
    with alive_bar(train_labels.shape[0]) as bar:
        for x, y in zip(train_sequences, train_labels):
            gradients = model_grad(x, jnp.array([y]), model_parameters)
            model_parameters = parameter_update(model_parameters, gradients)
            bar()
    print("Calculating accuracy")
    train_acc.append(np.mean([accuracy(jnp.array(x), jnp.array(y), model_parameters) for x, y in zip(mit.chunked(train_sequences, 10), mit.chunked(train_labels, 10))]))
    test_acc.append(np.mean([accuracy(jnp.array(x), jnp.array(y), model_parameters) for x, y in zip(mit.chunked(test_sequences, 10), mit.chunked(test_labels, 10))]))

    print(f"Train Accuracy: {train_acc[i]}")
    print(f"Test Accuracy: {test_acc[i]}")
    a = train_acc[i]


# %%
# Plot the train accuracy
import matplotlib.pyplot as plt

plt.plot(train_acc)
plt.plot(test_acc)
plt.show()


