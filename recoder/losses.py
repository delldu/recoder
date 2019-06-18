import torch
from torch import nn
import torch.nn.functional as F


def _reduce(x, reduction='elementwise_mean'):
  if reduction is 'none':
    return x
  elif reduction is 'elementwise_mean':
    return x.mean()
  elif reduction is 'sum':
    return x.sum()
  elif reduction is 'batchsize_mean':
    return x.sum() / x.shape[0]
  else:
    raise ValueError('No such reduction {} defined'.format(reduction))


class MSELoss(nn.Module):
  """
  Computes the weighted mean squared error loss.

  The weight for an observation x:

  .. math::
    w = 1 + confidence \\times x

  and the loss is:

  .. math::
    \ell(x, y) = w \cdot (y - x)^2

  Args:
    confidence (float, optional): the weighting of positive observations.
    reduction (string, optional): Specifies the reduction to apply to the output:
        'none' | 'elementwise_mean' | 'sum'. 'none': no reduction will be applied,
        'elementwise_mean': the sum of the output will be divided by the number of
        elements in the output, 'sum': the output will be summed. Default: 'elementwise_mean'
  """

  def __init__(self, confidence=0, reduction='elementwise_mean'):
    super(MSELoss, self).__init__()
    self.reduction = reduction
    self.confidence = confidence

  def forward(self, input, target):
    weights = 1 + self.confidence * (target > 0).float()
    loss = F.mse_loss(input, target, reduction='none')
    weighted_loss = weights * loss
    return _reduce(weighted_loss, reduction=self.reduction)


class MultinomialNLLLoss(nn.Module):
  """
  Computes the negative log-likelihood of the multinomial distribution.

  .. math::
    \ell(x, y) = \begin{cases}
      - y \cdot \log(softmax(x)), & \text{if apply\_softmax} = \text{True},\\
      - y \cdot \log(x),  & \text{if apply\_softmax} = \text{False}.
    \end{cases}

  Args:
    reduction (string, optional): Specifies the reduction to apply to the output:
        'none' | 'elementwise_mean' | 'sum'. 'none': no reduction will be applied,
        'elementwise_mean': the sum of the output will be divided by the number of
        elements in the output, 'sum': the output will be summed. Default: 'elementwise_mean'
    apply_softmax (bool, optional): Whether to apply softmax on the input or not. If False,
      the input should be the output of a softmax
  """

  def __init__(self, reduction='elementwise_mean',
               apply_softmax=True):
    super(MultinomialNLLLoss, self).__init__()
    self.reduction = reduction
    self.apply_softmax = apply_softmax

  def forward(self, input, target):

    if self.apply_softmax:
      loss = - target * F.log_softmax(input, dim=1)
    else:
      loss = - target * torch.log(input)

    return _reduce(loss, reduction=self.reduction)
