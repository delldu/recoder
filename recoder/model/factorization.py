import glog as log

import torch

import numpy as np

from recoder.model.base import BaseRecoder
from recoder.data.factorization import RecommendationDataset, RecommendationDataLoader, BatchCollator
from recoder.evaluation import RecommenderEvaluator
from recoder.nn.factorization import FactorizationModel


class Recoder(BaseRecoder):
  """
  Module to train/evaluate a recommendation :class:`recoder.nn.FactorizationModel`.

  Args:
    model (FactorizationModel): the factorization model to train.
    num_items (int, optional): the number of items to represent. If None, it will
      be computed from the first training dataset passed to ``train()``.
    num_users (int, optional): the number of users to represent. If not provided, it will
      be computed from the first training dataset passed to ``train()``.
    optimizer_type (str, optional): optimizer type (one of 'sgd', 'adam', 'adagrad', 'rmsprop').
    loss (str or torch.nn.Module, optional): loss function used to train the model.
      If loss is a ``str``, it should be `mse` for ``recoder.losses.MSELoss``, `logistic` for
      ``torch.nn.BCEWithLogitsLoss``, or `logloss` for ``recoder.losses.MultinomialNLLLoss``. If ``loss``
      is a ``torch.nn.Module``, then that Module will be used as a loss function and make sure that
      the loss reduction is a sum reduction and not an elementwise mean.
    use_cuda (bool, optional): use GPU when training/evaluating the model.
    user_based (bool, optional): If your model is based on users or not. If True, an exception will
      will be raised when there are inconsistencies between the users represented in the model
      and the users in the training datasets.
    item_based (bool, optional): If your model is based on items or not. If True, an exception will
      will be raised when there are inconsistencies between the items represented in the model
      and the items in the training datasets.
  """

  def __init__(self, model: FactorizationModel,
               num_items=None, num_users=None,
               optimizer_type='sgd', loss='mse',
               use_cuda=False, user_based=True,
               item_based=True):

    super().__init__(optimizer_type=optimizer_type, loss=loss,
                     use_cuda=use_cuda)

    self.model = model
    self.num_items = num_items
    self.num_users = num_users
    self.user_based = user_based
    self.item_based = item_based

    self.items = None
    self.users = None

  def _init_model(self):

    self.model.init_model(self.num_items, self.num_users)
    self.model = self.model.to(device=self.device)
    self.__model_initialized = True

  def _init_training(self, train_dataset):

    if self.items is None:
      self.items = train_dataset.items
    else:
      self.items = np.unique(np.append(self.items, train_dataset.items))

    if self.users is None:
      self.users = train_dataset.users
    else:
      self.users = np.unique(np.append(self.users, train_dataset.users))

    if self.item_based and self.num_items is None:
      self.num_items = int(np.max(self.items)) + 1
    elif self.item_based:
      assert self.num_items >= int(np.max(self.items)) + 1,\
        'The largest item id should be smaller than number of items.' \
        'If your model is not based on items, set item_based to False in Recoder constructor.'

    if self.user_based and self.num_users is None:
      self.num_users = int(np.max(self.users)) + 1
    elif self.user_based:
      assert self.num_users >= int(np.max(self.users)) + 1,\
        'The largest user id should be smaller than number of users.' \
        'If your model is not based on users, set user_based to False in Recoder constructor.'

  def _state(self):
    return {
      'items': self.items,
      'users': self.users,
      'num_items': self.num_items,
      'num_users': self.num_users
    }

  def _init_loaded_state(self, state):
    self.items = state['items']
    self.users = state['users']
    self.num_items = state['num_items']
    self.num_users = state['num_users']

  def train(self, train_dataset, val_dataset=None,
            lr=0.001, weight_decay=0, num_epochs=1,
            lr_momentum=0.9, lr_gamma=0,
            iters_per_epoch=None, batch_size=64, lr_milestones=None,
            negative_sampling=False, num_sampling_users=0,
            downsampling_threshold=None, num_data_workers=0,
            model_checkpoint_prefix=None, checkpoint_freq=0,
            eval_freq=0, eval_num_recommendations=None,
            eval_num_users=None, metrics=None,
            eval_input_split=0.5):
    """
    Trains the model

    Args:
      train_dataset (RecommendationDataset): train dataset.
      val_dataset (RecommendationDataset, optional): validation dataset.
      lr (float, optional): learning rate.
      weight_decay (float, optional): weight decay (L2 normalization).
      lr_momentum (float, optional): learning rate momentum.
      num_epochs (int, optional): number of epochs to train the model.
      iters_per_epoch (int, optional): number of training iterations per training epoch. If None,
        one epoch is full number of training samples in the dataset
      batch_size (int, optional): batch size
      lr_milestones (list, optional): optimizer learning rate epochs milestones.
      lr_gamma (float, optional): optimizer learning rate decay rate.
      negative_sampling (bool, optional): whether to apply mini-batch based negative sampling or not.
      num_sampling_users (int, optional): number of users to consider for sampling items.
        This is useful for increasing the number of negative samples in mini-batch based negative
        sampling while keeping the batch-size small. If 0, then num_sampling_users will
        be equal to batch_size.
      downsampling_threshold (float, optional): downsampling threshold (same downsampling method used in word2vec).
      num_data_workers (int, optional): number of data workers to use for building the mini-batches.
      checkpoint_freq (int, optional): epochs frequency of saving a checkpoint of the model
      model_checkpoint_prefix (str, optional): model checkpoint save path prefix
      eval_freq (int, optional): epochs frequency of doing an evaluation
      eval_num_recommendations (int, optional): num of recommendations to generate on evaluation
      eval_num_users (int, optional): number of users from the validation dataset to use for evaluation.
        If None, all users in the validation dataset are used for evaluation.
      metrics (list[Metric], optional): list of ``Metric`` used to evaluate the model
      eval_input_split (float, optional): the split percentage of the input to use as user history, in evaluation,
        and the remaining split as the user future interactions.
    """
    log.info('{} Mode'.format('CPU' if self.device.type == 'cpu' else 'GPU'))
    model_params = self.model.model_params()
    for param in model_params:
      log.info('Model {}: {}'.format(param, model_params[param]))
    log.info('Initial Learning Rate: {}'.format(lr))
    log.info('Weight decay: {}'.format(weight_decay))
    log.info('Batch Size: {}'.format(batch_size))
    log.info('Optimizer: {}'.format(self.optimizer_type))
    log.info('LR milestones: {}'.format(lr_milestones))
    log.info('Loss Function: {}'.format(self.loss))

    self._init_training(train_dataset)

    if num_sampling_users == 0:
      num_sampling_users = batch_size

    assert num_sampling_users >= batch_size and num_sampling_users % batch_size == 0, \
      "number of sampling users should be a multiple of the batch size"

    train_dataloader = RecommendationDataLoader(train_dataset, batch_size=batch_size,
                                                negative_sampling=negative_sampling,
                                                num_sampling_users=num_sampling_users,
                                                downsampling_threshold=downsampling_threshold,
                                                num_workers=num_data_workers)
    if val_dataset is not None:
      val_dataloader = RecommendationDataLoader(val_dataset, batch_size=batch_size,
                                                negative_sampling=negative_sampling,
                                                num_sampling_users=num_sampling_users,
                                                num_workers=num_data_workers)
    else:
      val_dataloader = None

    self.train_loop(lr=lr,
                    weight_decay=weight_decay,
                    train_dataloader=train_dataloader,
                    val_dataloader=val_dataloader,
                    num_epochs=num_epochs,
                    current_epoch=self.current_epoch,
                    lr_milestones=lr_milestones,
                    batch_size=batch_size,
                    model_checkpoint_prefix=model_checkpoint_prefix,
                    checkpoint_freq=checkpoint_freq,
                    eval_freq=eval_freq,
                    metrics=metrics,
                    num_recommendations=eval_num_recommendations,
                    iters_per_epoch=iters_per_epoch,
                    num_users=eval_num_users,
                    input_split=eval_input_split,
                    lr_gamma=lr_gamma,
                    lr_momentum=lr_momentum,
                    eval_dataset=val_dataset)

  def predict(self, users_interactions, return_input=False):
    """
    Predicts the user interactions with all items

    Args:
      users_interactions (UsersInteractions): A batch of users' history consisting of list of ``Interaction``
      return_input (bool, optional): whether to return the dense input batch

    Returns:
      if ``return_input`` is ``True`` a tuple of the predictions and the input batch
      is returned, otherwise only the predictions are returned
    """
    if self.model is None:
      raise Exception('Model not initialized.')

    self.model.eval()

    batch_collator = BatchCollator(batch_size=len(users_interactions.users), negative_sampling=False)

    input = batch_collator.collate(users_interactions)
    batch = input[0]
    batch.to(device=self.device)

    model_output = self.model(batch)
    input_dense = batch.sparse_tensor.to_dense()

    return model_output.output, input_dense if return_input else model_output.output

  def recommend(self, users_interactions, num_recommendations):
    """
    Generate list of recommendations for each user in ``users_hist``.

    Args:
      users_interactions (UsersInteractions): list of users interactions.
      num_recommendations (int): number of recommendations to generate.

    Returns:
      list: list of recommended items for each user in users_interactions.
    """
    output, input = self.predict(users_interactions, return_input=True)
    # Set input items output to -inf so that they don't get recommended
    output[input > 0] = - float('inf')

    top_output, top_ind = torch.topk(output, num_recommendations, dim=1, sorted=True)

    recommendations = top_ind.tolist()

    return recommendations

  def evaluate(self, eval_dataset, num_recommendations, metrics, batch_size=1,
               num_users=None, input_split=0.5):
    """
    Evaluates the current model given an evaluation dataset.

    Args:
      eval_dataset (RecommendationDataset): evaluation dataset
      num_recommendations (int): number of top recommendations to consider.
      metrics (list): list of ``Metric`` to use for evaluation.
      batch_size (int, optional): batch size of computations.
      num_users (int, optional): the number of users from the dataset to evaluate on. If None,
        evaluate on all users.
      input_split (float, optional): the split percentage of the input to use as user history,
        and the remaining split as the user future interactions.
    """
    if self.model is None:
      raise Exception('Model not initialized')

    self.model.eval()

    evaluator = RecommenderEvaluator(self, metrics)

    results = evaluator.evaluate(eval_dataset, batch_size=batch_size, num_users=num_users,
                                 input_split=input_split, num_recommendations=num_recommendations)
    return results
