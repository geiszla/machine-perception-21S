"""
Some loss functions.

Copyright ETH Zurich, Manuel Kaufmann
"""


def mse(predictions, targets):
    """
    Compute the MSE.

    :param predictions: A tensor of shape (N, MAX_SEQ_LEN, -1)
    :param targets: A tensor of shape (N, MAX_SEQ_LEN, -1)
    :return: The MSE between predictions and targets.
    """
    diff = predictions - targets
    loss_per_sample_and_seq = (diff * diff).sum(dim=-1)  # (N, F)
    return loss_per_sample_and_seq.mean()


def l1_loss(predictions, targets):
    """
    Compute the L1-loss.

    :param predictions: A tensor of shape (N, MAX_SEQ_LEN, -1)
    :param targets: A tensor of shape (N, MAX_SEQ_LEN, -1)
    :return: The L1-loss between predictions and targets.
    """
    diff = (predictions - targets).abs()
    return diff.sum(dim=-1).mean()
