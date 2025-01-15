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

from .core import * 


def model_forward2(input_sequence, parameters, prob, key, training = True):
    """
    The model forward function, which takes in the input sequence and the parameters and returns the output of the model.

    Parameters = Linear_encoder_parameter,  LRU, seconday_parameters, Linear_decoder_parameter 
    """
    
    Linear_encoder_parameter,  LRU, seconday_parameters, Linear_decoder_parameter = parameters

    x = forward_mlp(Linear_encoder_parameter, input_sequence)
    skip = x
    x = layer_normalization_sequence(x)
    x = forward_LRU(LRU, x)
    x = forward_mlp_with_dropout(seconday_parameters, x, key, prob, training) + skip
    x = max_pooling(x)
    x = forward_mlp_linear_with_classification(Linear_decoder_parameter, x)

    return x


# Batch model forward
batch_model_forward2 = vmap(model_forward2, in_axes=(0, None, None, None, None))


@jit
def loss_fn2(input_sequences, target_sequences, parameters, prob, key, training):
    y = batch_model_forward2(input_sequences, parameters, prob, key, training)

    # Binary cross entropy loss
    return -jnp.mean(jnp.sum(target_sequences * jnp.log(y), axis=1))

@jit
def model_grad2(input_sequence, target_sequence, parameters, prob, key, training):
    return grad(loss_fn2, argnums=2)(input_sequence, target_sequence, parameters, prob, key, training)


@jit
def accuracy2(input_sequences, target_sequences, parameters, prob, key, training):
    """
    Perfrom a batch accuracy measuremet

    input_sequences = [waveform1, waveform2] where waveform.shape = (len(waveform, 1))
    target_sequences = [one_hot(int), one_hot(int), one_hot(int)]
    parameters = model parameters
    """
    y = batch_model_forward2(input_sequences, parameters, prob, key, training)
    return jnp.mean(jnp.argmax(y, axis=1) == jnp.argmax(target_sequences, axis=1))

batch_model_grad2 = vmap(model_grad2, in_axes=(0, 0, None, None, None, None))

