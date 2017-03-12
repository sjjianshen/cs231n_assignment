import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W)
  dscores = np.zeros_like(scores)
  sample_size = scores.shape[0]
  for i in np.arange(sample_size):
      exp_scores = np.exp(scores[i,:])
      p = exp_scores / np.sum(exp_scores)
      loss += -np.log(p[y[i]])
      dscores[i,:] = p
      dscores[i,y[i]] -= 1
  loss /= sample_size
  loss += 0.5 * reg * np.sum(W ** 2)
  dscores /= sample_size
  dW = X.T.dot(dscores)
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  sample_size = X.shape[0]
  scores = X.dot(W)
  exp_score = np.exp(scores)
  probs = exp_score / np.sum(exp_score, axis=1, keepdims=True)
  loss = np.sum(-np.log(probs[np.arange(sample_size), y]))
  loss /= sample_size
  loss += 0.5 * reg * np.sum(W ** 2)
  dscores = probs.copy()
  dscores[np.arange(sample_size), y] -= 1
  dW = X.T.dot(dscores)
  dW /= sample_size
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

