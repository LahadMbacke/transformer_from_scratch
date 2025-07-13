import numpy as np
import math


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=-1)

def self_attention(Q, K, V,mask=None):
    d_k = Q.shape[-1]  # Dimension of keys/values
    produit_QK = np.matmul(Q, K.T)  # Compute Q * K^T

    if mask is not None:
        scaled = ((produit_QK)/ math.sqrt(d_k))+ mask
    else:
        scaled = produit_QK / math.sqrt(d_k)
    attention_weights = softmax(scaled)  # Apply softmax to get attention weights
    output = np.matmul(attention_weights, V)  # Compute the final output
    return output, attention_weights