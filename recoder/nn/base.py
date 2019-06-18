from torch import nn


class ModelOutput:
  """
  Represents the output of a forward propagation through a :class:`BaseModel`.

  Args:
    output (torch.Tensor, optional): the model output (predictions).
    target (torch.Tensor, optional): the target predictions.
    weight (torch.Tensor, optional): the weight to multiply loss with.
    loss (torch.Tensor, optional): the loss value.
  """
  def __init__(self, output=None, target=None,
               weight=None, loss=None):
    self.output = output
    self.target = target
    self.weight = weight
    self.loss = loss


class BaseModel(nn.Module):

  def forward(self, *args, **kwargs):
    """
    Forward pass through the model

    Returns:
      ModelOutput: model output
    """
    raise NotImplementedError
