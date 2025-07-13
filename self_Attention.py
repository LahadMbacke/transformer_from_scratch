import numpy as np
import math




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


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=-1)









L,d_k,d_v = 5,8,8  # Sequence length, dimension of keys/values

Q = np.random.rand(L, d_k)  # Query matrix
K = np.random.rand(L, d_k)  # Key matrix
V = np.random.rand(L, d_v)  # Value matrix

# Mask to prevent attention to future tokens
mask = np.tril(np.ones((L,L)))
mask[mask == 0] = -np.inf  # Set upper triangle to -inf
mask[mask == 1] = 0

print("Attention output shape:", self_attention(Q, K, V, mask))