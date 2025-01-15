import numpy as np

import jax
import jax.numpy as jnp
from jax import jit, grad, vmap
import optax


import pickle as pkl

import itertools
import more_itertools as mit

import os

import gc

parallel_scan = jax.lax.associative_scan

def binary_operator_diag(element_i, element_j):
    a_i, bu_i = element_i
    a_j, bu_j = element_j

    return a_j * a_i, a_j * bu_i + bu_j


def init_lru_parameters(N, H, r_min = 0.0, r_max = 1, max_phase = 6.28):
    """
    Initialize the LRU parameters

    N : integer, the state dimension (Memory)
    H : integer, the model dimension (Output dimension and input)
    r_min : float, the minimum value of the radius of the complex number
    r_max : float, the maximum value of the radius of the complex number
    max_phase : float, the maximum value of the phase of the complex number

    return : tuple, the LRU parameters

    """
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
    #print(B_norm.shape)
    C = C_re + 1j*C_im

    Lambda_elements = jnp.repeat(Lambda[None, ...], input_sequence.shape[0], axis=0)

    Bu_elements = jax.vmap(lambda u: B_norm @ u)(input_sequence)
    elements = (Lambda_elements, Bu_elements)
    _, inner_states = parallel_scan(binary_operator_diag, elements) # all x_k
    y = jax.vmap(lambda x, u: (C @ x).real + D * u)(inner_states, input_sequence)


    return y

def init_mlp_parameters(layers):
    """
    Intialize the parameters of the MLP

    layers : list of integers, where each element is the number of neurons in each layer

    return : list of tuples, where each tuple is the weight matrix and bias vector of the layer
    """
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

@jit
def dropout(input, prob, random_key):
    """
    Single layer dropout function
    """

    keep_prob = 1.0 - prob
    mask = jax.random.bernoulli(random_key, keep_prob, input.shape)

    return jnp.where(mask, input, 0)

@jit
def forward_mlp_with_dropout(mlp_parameters, input, random_key, prob, Training = True, activation_function = jnp.tanh):
    # Forward pass of the MLP with dropout
    
    x = input


    for W, b in mlp_parameters:
        x = dropout(x, prob, random_key)
        x = x @ W + b
        x = activation_function(x)
        random_key, _ = jax.random.split(random_key)
    return x


def forward_mlp_linear_with_classification(mlp_parameters, input, activation_function = jnp.tanh):
    
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



def layer_normalization(activations):
    mu  = jnp.mean(activations)
    sigma = jnp.std(activations)
    return (activations - mu) / sigma

layer_normalization_sequence = vmap(layer_normalization)

def max_pooling(sequence_to_pool):
    return jnp.max(sequence_to_pool, axis=0)

def mean_pooling(sequence_to_pool):
    return jnp.mean(sequence_to_pool, axis=0)

def sum_pooling(sequence_to_pool):
    return jnp.sum(sequence_to_pool, axis=0)

def model_forward(input_sequence, parameters, training = True):
    """
    The model forward function, which takes in the input sequence and the parameters and returns the output of the model.

    Parameters = Linear_encoder_parameter,  LRU, seconday_parameters, Linear_decoder_parameter 
    """
    Linear_encoder_parameter,  LRU, seconday_parameters, Linear_decoder_parameter = parameters

    x = forward_mlp(Linear_encoder_parameter, input_sequence)
    skip = x
    x = layer_normalization_sequence(x)
    x = forward_LRU(LRU, x)
    x = forward_mlp(seconday_parameters, x) + skip
    x = max_pooling(x)
    x = forward_mlp_linear_with_classification(Linear_decoder_parameter, x)

    return x


# Batch model forward
batch_model_forward = vmap(model_forward, in_axes=(0, None))

def one_hot(x, k, dtype=jnp.float32):
  """Create a one-hot encoding of x of size k."""
  return jnp.array(x[:, None] == jnp.arange(k), dtype)

@jit
def loss_fn(input_sequences, target_sequences, parameters):
    y = batch_model_forward(input_sequences, parameters)

    # Binary cross entropy loss
    return -jnp.mean(jnp.sum(target_sequences * jnp.log(y), axis=1))

@jit
def model_grad(input_sequence, target_sequence, parameters):
    return grad(loss_fn, argnums=2)(input_sequence, target_sequence, parameters)


@jit
def accuracy(input_sequences, target_sequences, parameters):
    """
    Perfrom a batch accuracy measuremet

    input_sequences = [waveform1, waveform2] where waveform.shape = (len(waveform, 1))
    target_sequences = [one_hot(int), one_hot(int), one_hot(int)]
    parameters = model parameters
    """
    y = batch_model_forward(input_sequences, parameters)
    return jnp.mean(jnp.argmax(y, axis=1) == jnp.argmax(target_sequences, axis=1))

batch_model_grad = vmap(model_grad, in_axes=(0, 0, None))



def load_data(data_file_path, batch_size, targets, test_ratio = 0.8):
    """
    Load the data from the data file path

    data structure = [(Wave_sequence, target).... (Wave_sequence, target)]

    batch_size : integer

    targets : integer

    test_ratio : float between (0, 1]

    return : train_sequences, train_labels, test_sequences, test_labels
    """
    with open(data_file_path, "rb") as f:
        data = pkl.load(f)


    # Permutate the order of the data
    perm = np.random.permutation(len(data))

    # Shuffle according to the permutation
    shuffled_data = [data[i] for i in perm]

    # make the data split
    train_sequences = jnp.array([x[0] for x in shuffled_data[:int(test_ratio*len(shuffled_data))]]).reshape((int(test_ratio*len(shuffled_data)),len(shuffled_data[0][0]),1))
    test_sequences = jnp.array([x[0] for x in shuffled_data[int(test_ratio*len(shuffled_data)):]]).reshape((len(shuffled_data) - int(test_ratio*len(shuffled_data)),len(shuffled_data[0][0]),1))

    train_labels = one_hot(jnp.array([x[1] for x in shuffled_data[:int(test_ratio*len(shuffled_data))]]), targets)
    test_labels = one_hot(jnp.array([x[1] for x in shuffled_data[int(test_ratio*len(shuffled_data)):]]), targets)

    # Batch the sequences
    train_sequences = train_sequences.reshape((int(test_ratio*len(shuffled_data)/batch_size),batch_size,-1,1))
    test_sequences = test_sequences.reshape((int((len(shuffled_data) - int(test_ratio*len(shuffled_data)))/batch_size),batch_size,-1,1))

    train_labels = train_labels.reshape((int(test_ratio*len(shuffled_data)/batch_size),batch_size,-1))
    test_labels = test_labels.reshape((int((len(shuffled_data) - int(test_ratio*len(shuffled_data)))/batch_size),batch_size,-1))

    return train_sequences, train_labels, test_sequences, test_labels

def load_data_stft(data_file_path, batch_size, targets, test_ratio = 0.8, window_filter = 'hann', length = 256, hop = 256//2):
    with open(data_file_path, "rb") as f:
        data=pkl.load(f)

    # Permutate the order of the data
    perm = np.random.permutation(len(data))

    shuffled_data = [data[i] for i in perm]

    # make the data split
    train_sequences = jnp.array([x[0] for x in shuffled_data[:int(test_ratio*len(shuffled_data))]])
    test_sequences = jnp.array([x[0] for x in shuffled_data[int(test_ratio*len(shuffled_data)):]])

    train_labels = one_hot(jnp.array([x[1] for x in shuffled_data[:int(test_ratio*len(shuffled_data))]]), targets)
    test_labels = one_hot(jnp.array([x[1] for x in shuffled_data[int(test_ratio*len(shuffled_data)):]]), targets)

    
    # Perform the stft of dataset
    train_sequences = jax.vmap(lambda input: jnp.abs(jnp.transpose(jax.scipy.signal.stft(input, fs=2000, window=window_filter, nperseg=length, noverlap = hop, return_onesided=True)[2])))(train_sequences)
    test_sequences = jax.vmap(lambda input: jnp.abs(jnp.transpose(jax.scipy.signal.stft(input, fs=2000, window=window_filter, nperseg=length, noverlap = hop, return_onesided=True)[2])))(test_sequences)


    # Batch the sequences
    train_sequences = train_sequences.reshape((int(test_ratio*len(shuffled_data)/batch_size),batch_size,train_sequences.shape[1],train_sequences.shape[2]))
    test_sequences = test_sequences.reshape((int((len(shuffled_data) - int(test_ratio*len(shuffled_data)))/batch_size),batch_size,test_sequences.shape[1],test_sequences.shape[2]))

    train_labels = train_labels.reshape((int(test_ratio*len(shuffled_data)/batch_size),batch_size,-1))
    test_labels = test_labels.reshape((int((len(shuffled_data) - int(test_ratio*len(shuffled_data)))/batch_size),batch_size,-1))

    return train_sequences, train_labels, test_sequences, test_labels

    


def init_lru_parameters_uneven(N, H_in, H_out, r_min = 0.0, r_max = 1, max_phase = 6.28):
    """
    Initialize the LRU parameters

    N : integer, the state dimension (Memory)
    H_in : integer, the model dimension (Input dimension)
    H_out : integer, the model dimensions output (Output Dimesion.)
    r_min : float, the minimum value of the radius of the complex number
    r_max : float, the maximum value of the radius of the complex number
    max_phase : float, the maximum value of the phase of the complex number

    return : tuple, the LRU parameters

    """
    # N: state dimension, H: model dimension
    # Initialization of Lambda is complex valued distributed uniformly on ring
    # between r_min and r_max, with phase in [0, max_phase].

    u1 = np.random.uniform(size = (N,))
    u2 = np.random.uniform(size = (N,))

    nu_log = np.log(-0.5*np.log(u1*(r_max**2-r_min**2) + r_min**2))
    theta_log = np.log(max_phase*u2)

    # Glorot initialized Input/Output projection matrices
    B_re = np.random.normal(size=(N,H_in))/np.sqrt(2*H_in)
    B_im = np.random.normal(size=(N,H_in))/np.sqrt(2*H_in)
    C_re = np.random.normal(size=(H_out,N))/np.sqrt(N)
    C_im = np.random.normal(size=(H_out,N))/np.sqrt(N)
    D = np.random.normal(size=(H_out,H_in))/np.sqrt(H_in)

    # Normalization
    diag_lambda = np.exp(-np.exp(nu_log) + 1j*np.exp(theta_log))
    gamma_log = np.log(np.sqrt(1-np.abs(diag_lambda)**2))

    return nu_log, theta_log, B_re, B_im, C_re, C_im, D, gamma_log

def forward_LRU_uneven(lru_parameters, input_sequence):
    # Unpack the LRU parameters
    nu_log, theta_log, B_re, B_im, C_re, C_im, D, gamma_log = lru_parameters

    # Initialize the hidden state
    Lambda = jnp.exp(-jnp.exp(nu_log) + 1j*jnp.exp(theta_log))
    B_norm = (B_re + 1j*B_im) * jnp.expand_dims(jnp.exp(gamma_log), axis=-1)
    #print(B_norm.shape)
    C = C_re + 1j*C_im

    Lambda_elements = jnp.repeat(Lambda[None, ...], input_sequence.shape[0], axis=0)

    Bu_elements = jax.vmap(lambda u: B_norm @ u)(input_sequence)
    elements = (Lambda_elements, Bu_elements)
    _, inner_states = parallel_scan(binary_operator_diag, elements) # all x_k
    y = jax.vmap(lambda x, u: (C @ x).real + D @ u )(inner_states, input_sequence)


    return y