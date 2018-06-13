import numpy as np


def softmax(logits, y):
    """
    Computes the loss and gradient for softmax classification.

    Args:
    - logits: A numpy array of shape (N, C)
    - y: A numpy array of shape (N,). y represents the labels corresponding to
    logits, where y[i] is the label of logits[i], and the value of y have a 
    range of 0 <= y[i] < C

    Returns (as a tuple):
    - loss: Loss scalar
    - dlogits: Loss gradient with respect to logits
    """
    # loss, dlogits = None, None
    """
    TODO: Compute the softmax loss and its gradient using no explicit loops
    Store the loss in loss and the gradient in dlogits. If you are not careful
    here, it is easy to run into numeric instability. Don't forget the
    normalisation!
    """
    ###########################################################################
    #                           BEGIN OF YOUR CODE                            #
    ###########################################################################
    K = -(logits.shape[1])
    #print(logits)

    N, C = logits.shape
    #print(N)
    soft_logits = np.exp(logits + K) / np.exp(logits + K).sum(axis=1)[:, None]

    neg_log_like = soft_logits[np.arange(N), y]
    #neg_log_like[neg_log_like == 0] = 1e-7
    #print(neg_log_like)

    if y is None:
        return soft_logits

    y_one_hot = np.zeros((N, C))
    y_one_hot[np.arange(N), y] = 1.

    loss = -1/N * np.sum(np.log(neg_log_like))

    dlogits = 1/N * (soft_logits - y_one_hot)
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
    return loss, dlogits
