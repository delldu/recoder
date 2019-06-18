import os

import glog as log

import numpy as np

import torch
import torch.optim as optim
from torch.nn import BCEWithLogitsLoss
from torch.optim.lr_scheduler import MultiStepLR, StepLR

from tqdm import tqdm

from recoder import __version__
from recoder.losses import MSELoss, MultinomialNLLLoss, _reduce


class BaseRecoder(object):
  """
  Base class to train and evaluate a recommendation model :class:`recoder.nn.base.BaseModel`.

  Args:
    optimizer_type (str, optional): optimizer type (one of 'sgd', 'adam', 'adagrad', 'rmsprop').
    loss (str or torch.nn.Module, optional): loss function used to train the model.
      If loss is a ``str``, it should be `mse` for mean square error, `logistic` for
      logistic loss, or `logloss` for softmax log loss. If ``loss``
      is a ``torch.nn.Module``, then that Module will be used as a loss function and
      you should handle the loss reduction by yourself.
    use_cuda (bool, optional): use GPU when training/evaluating the model.
  """

  def __init__(self, optimizer_type='sgd',
               loss='mse', use_cuda=False):

    self.optimizer_type = optimizer_type
    self.loss = loss
    self.use_cuda = use_cuda

    if self.use_cuda:
      self.device = torch.device('cuda')
    else:
      self.device = torch.device('cpu')

    self.model = None
    self.optimizer = None
    self.sparse_optimizer = None
    self.current_epoch = 1
    self._model_initialized = False
    self.__optimizer_state_dict = None
    self.__sparse_optimizer_state_dict = None
    self._loss_reduction = 'batchsize_mean'

  def __init_model(self):
    if self._model_initialized:
      return

    self._init_model()
    self._model_initialized = True

  def _init_model(self):
    raise NotImplementedError

  def _init_loss_module(self):
    if self.loss is None:
      raise ValueError('No loss function defined')

    if issubclass(self.loss.__class__, torch.nn.Module):
      self.loss_module = self.loss
      self._loss_reduction = 'none'
      return

    loss_map = {
      'logistic': BCEWithLogitsLoss(reduction='none'),
      'mse': MSELoss(reduction='none'),
      'logloss': MultinomialNLLLoss(reduction='none')
    }

    if self.loss not in loss_map:
      raise ValueError('Unknown loss function {}'.format(self.loss))

    self.loss_module = loss_map[self.loss]

  def _init_optimizer(self, lr, weight_decay, lr_momentum):
    # When continuing training on the same Recoder instance
    if self.optimizer is not None:
      self.__optimizer_state_dict = self.optimizer.state_dict()

    if self.sparse_optimizer is not None:
      self.__sparse_optimizer_state_dict = self.sparse_optimizer.state_dict()

    # Collecting sparse parameter names
    sparse_params_names = []
    sparse_modules = [torch.nn.Embedding, torch.nn.EmbeddingBag]
    for module_name, module in self.model.named_modules():
      if type(module) in sparse_modules and module.sparse:
        sparse_params_names.extend([module_name + '.' + param_name
                                    for param_name, param in module.named_parameters()])

    # Initializing optimizer params
    params = []
    sparse_params = []
    for param_name, param in self.model.named_parameters():
      _weight_decay = weight_decay

      if 'bias' in param_name:
        _weight_decay = 0

      if param_name in sparse_params_names:
        # If module is an embedding layer with sparse gradients
        # then add its parameters to sparse optimizer
        sparse_params.append({'params': param, 'weight_decay': _weight_decay})
      else:
        params.append({'params': param, 'weight_decay': _weight_decay})

    if self.optimizer_type == "adam":
      if len(params) > 0:
        self.optimizer = optim.Adam(params, lr=lr)

      if len(sparse_params) > 0:
        self.sparse_optimizer = optim.SparseAdam(sparse_params, lr=lr)

    elif self.optimizer_type == "adagrad":
      if len(sparse_params) > 0:
        raise ValueError('Sparse gradients optimization not supported with adagrad')

      self.optimizer = optim.Adagrad(params, lr=lr)
    elif self.optimizer_type == "sgd":

      if len(params) > 0:
        self.optimizer = optim.SGD(params, lr=lr, momentum=lr_momentum)

      if len(sparse_params) > 0:
        self.sparse_optimizer = optim.SGD(sparse_params, lr=lr, momentum=lr_momentum)

    elif self.optimizer_type == "rmsprop":
      if len(sparse_params) > 0:
        raise ValueError('Sparse gradients optimization not supported with rmsprop')

      self.optimizer = optim.RMSprop(params, lr=lr, momentum=lr_momentum)
    else:
      raise Exception('Unknown optimizer kind')

    if self.__optimizer_state_dict is not None:
      self.optimizer.load_state_dict(self.__optimizer_state_dict)
      self.__optimizer_state_dict = None # no need for this anymore

    if self.__sparse_optimizer_state_dict is not None and self.sparse_optimizer is not None:
      self.sparse_optimizer.load_state_dict(self.__sparse_optimizer_state_dict)
      self.__sparse_optimizer_state_dict = None

    assert self.optimizer is not None or self.sparse_optimizer is not None, "No optimizer was initialized"

  def _init_loaded_state(self, state):
    return

  def init_from_model_file(self, model_file):
    """
    Initializes the model from a pre-trained model

    Args:
       model_file (str): the pre-trained model file path
    """
    log.info('Loading model from: {}'.format(model_file))
    if not os.path.isfile(model_file):
      raise Exception('No state file found in {}'.format(model_file))
    model_saved_state = torch.load(model_file, map_location='cpu')
    model_params = model_saved_state['model_params']
    self.current_epoch = model_saved_state['last_epoch']
    self.loss = model_saved_state.get('loss', self.loss)
    self.optimizer_type = model_saved_state['optimizer_type']
    self.__optimizer_state_dict = model_saved_state.get('optimizer', None)
    self.__sparse_optimizer_state_dict = model_saved_state.get('sparse_optimizer', None)
    self._init_loaded_state(model_saved_state)

    self.model.load_model_params(model_params)
    self._init_model()
    self.model.load_state_dict(model_saved_state['model'])

  def _state(self):
    return {}

  def save_state(self, model_checkpoint_prefix):
    """
    Saves the model state in the path starting with ``model_checkpoint_prefix`` and appending it
    with the model current training epoch

    Args:
      model_checkpoint_prefix (str): the model save path prefix

    Returns:
      the model state file path
    """
    checkpoint_file = "{}_epoch_{}.model".format(model_checkpoint_prefix, self.current_epoch)
    log.info("Saving model to {}".format(checkpoint_file))
    current_state = {
      'recoder_version': __version__,
      'model_params': self.model.model_params(),
      'last_epoch': self.current_epoch,
      'model': self.model.state_dict(),
      'optimizer_type': self.optimizer_type,
    }

    current_state.update(self._state())

    if self.optimizer is not None:
      current_state['optimizer'] = self.optimizer.state_dict()

    if self.sparse_optimizer is not None:
      current_state['sparse_optimizer'] = self.sparse_optimizer.state_dict()

    if type(self.loss) is str:
      current_state['loss'] = self.loss

    torch.save(current_state, checkpoint_file)
    return checkpoint_file

  def train(self, *args, **kwargs):
    raise NotImplementedError

  def train_loop(self, lr, weight_decay, lr_momentum, train_dataloader,
                 val_dataloader, num_epochs, current_epoch, lr_milestones,
                 lr_gamma, model_checkpoint_prefix, checkpoint_freq,
                 eval_freq, iters_per_epoch, **eval_args):

    self._init_model()
    self._init_optimizer(lr=lr, weight_decay=weight_decay, lr_momentum=lr_momentum)
    self._init_loss_module()

    lr_scheduler, sparse_lr_scheduler = None, None
    if lr_milestones is not None:
      _last_epoch = -1 if self.current_epoch == 1 else (self.current_epoch - 2)
      if self.optimizer is not None:
        lr_scheduler = MultiStepLR(self.optimizer, milestones=lr_milestones,
                                   gamma=lr_gamma, last_epoch=_last_epoch)
      if self.sparse_optimizer is not None:
        sparse_lr_scheduler = MultiStepLR(self.sparse_optimizer, milestones=lr_milestones,
                                          gamma=lr_gamma, last_epoch=_last_epoch)
    elif lr_gamma > 0:
      _last_epoch = -1 if self.current_epoch == 1 else (self.current_epoch - 2)
      if self.optimizer is not None:
        lr_scheduler = StepLR(self.optimizer, step_size=1,
                              gamma=lr_gamma, last_epoch=_last_epoch)
      if self.sparse_optimizer is not None:
        sparse_lr_scheduler = StepLR(self.sparse_optimizer, step_size=1,
                                     gamma=lr_gamma, last_epoch=_last_epoch)
    else:
      lr_scheduler = None

    num_batches = len(train_dataloader)

    iters_processed = 0
    if iters_per_epoch is None:
      iters_per_epoch = num_batches

    for epoch in range(current_epoch, num_epochs + 1):
      self.current_epoch = epoch
      self.model.train()
      aggregated_losses = []

      if lr_scheduler is not None or sparse_lr_scheduler is not None:
        if lr_scheduler is not None:
          lr_scheduler.step()
          lr = lr_scheduler.get_lr()[0]

        if sparse_lr_scheduler is not None:
          sparse_lr_scheduler.step()
          lr = sparse_lr_scheduler.get_lr()[0]
      else:
        if self.optimizer is not None:
          lr = self.optimizer.defaults['lr']
        else:
          lr = self.sparse_optimizer.defaults['lr']

      description = 'Epoch {}/{}'.format(epoch, num_epochs)

      if iters_processed == 0 or iters_processed == num_batches:
        # If we are starting from scratch,
        # or we iterated through the whole dataloader,
        # reset and reinitialize the iterator
        iters_processed = 0
        iterator = enumerate(train_dataloader, 1)

      iters_to_process = min(iters_per_epoch, num_batches - iters_processed)
      iters_processed += iters_to_process

      progress_bar = tqdm(range(iters_to_process), desc=description)

      for batch_itr, input in iterator:

        if self.optimizer is not None:
          self.optimizer.zero_grad()

        if self.sparse_optimizer is not None:
          self.sparse_optimizer.zero_grad()

        loss = self.__compute_loss(input)

        loss.backward()
        if self.optimizer is not None:
          self.optimizer.step()

        if self.sparse_optimizer is not None:
          self.sparse_optimizer.step()

        aggregated_losses.append(loss.item())

        progress_bar.set_postfix(loss=aggregated_losses[-1],
                                 lr=lr,
                                 refresh=False)

        progress_bar.update()

        if batch_itr % iters_per_epoch == 0:
          break

      postfix = {'loss': np.mean(aggregated_losses), 'lr': lr}
      if val_dataloader is not None:
        val_loss = self._validate(val_dataloader)
        postfix['val_loss'] = val_loss
        if eval_args and eval_freq > 0 and epoch % eval_freq == 0:
          results = self.evaluate(**eval_args)
          for metric in results:
            postfix[str(metric)] = np.mean(results[metric])

      progress_bar.set_postfix(postfix)
      progress_bar.close()

      if model_checkpoint_prefix and \
          ((checkpoint_freq > 0 and epoch % checkpoint_freq == 0) or epoch == num_epochs):
        self.save_state(model_checkpoint_prefix)

  def _validate(self, val_dataloader):
    self.model.eval()

    total_loss = 0.0
    num_batches = 1

    for itr, batch in enumerate(val_dataloader):
      loss = self.__compute_loss(batch)
      total_loss += loss.item()
      num_batches = itr + 1

    avg_loss = total_loss / num_batches

    return avg_loss

  def __compute_loss(self, input):

    input.to(self.device)

    model_output = self.model(input)  # type: ModelOutput

    if model_output.loss is not None:
      return model_output.loss

    loss = self.loss_module(model_output.output, model_output.target)

    if model_output.weight is not None:
      loss = loss * model_output.weight

    loss = _reduce(loss, reduction=self._loss_reduction)

    return loss

  def predict(self, *args, **kwargs):
    raise NotImplementedError

  def recommend(self, *args, **kwargs):
    raise NotImplementedError

  def evaluate(self, *args, **kwargs):
    raise NotImplementedError
