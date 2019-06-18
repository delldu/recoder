import glog as log

import torch

import numpy as np

from recoder.model.base import BaseRecoder
from recoder.data.sequence import SequenceDataset, SequenceDataLoader, SequencesCollator, Sequences
from recoder.evaluation import SequentialRecommenderEvaluator
from recoder.nn.sequence import SequenceModel


class SequenceRecoder(BaseRecoder):
  """
  Module to train/evaluate a recommendation :class:`recoder.nn.SequenceModel`.

  Args:
    model (SequenceModel): the sequence model to train.
    num_items (int, optional): the number of items to represent. If None, it will
      be computed from the first training dataset passed to ``train()``.
    num_sequences (int, optional): the number of sequences to represent. If not provided, it will
      be computed from the first training dataset passed to ``train()``.
    optimizer_type (str, optional): optimizer type (one of 'sgd', 'adam', 'adagrad', 'rmsprop').
    loss (str or torch.nn.Module, optional): loss function used to train the model.
      If loss is a ``str``, it should be `mse` for ``recoder.losses.MSELoss``, `logistic` for
      ``torch.nn.BCEWithLogitsLoss``, or `logloss` for ``recoder.losses.MultinomialNLLLoss``. If ``loss``
      is a ``torch.nn.Module``, then that Module will be used as a loss function and make sure that
      the loss reduction is a sum reduction and not an elementwise mean.
    use_cuda (bool, optional): use GPU when training/evaluating the model.
    sequence_based (bool, optional): If your model models sequences or not. If True, an exception will
      will be raised when there are inconsistencies between the sequences represented in the model
      and the sequences in the training datasets.
    item_based (bool, optional): If your model is based on items or not. If True, an exception will
      will be raised when there are inconsistencies between the items represented in the model
      and the items in the training datasets.
  """

  def __init__(self, model: SequenceModel,
               num_items=None, num_sequences=None,
               optimizer_type='sgd', loss='mse',
               use_cuda=False, sequence_based=False,
               item_based=True):

    super().__init__(optimizer_type=optimizer_type, loss=loss,
                     use_cuda=use_cuda)

    self.model = model
    self.num_items = num_items
    self.num_sequences = num_sequences
    self.sequence_based = sequence_based
    self.item_based = item_based

    self.items = None
    self.sequence_ids = None

  def _init_model(self):

    self.model.init_model(self.num_items, self.num_sequences)
    self.model = self.model.to(device=self.device)
    self.__model_initialized = True

  def _init_training(self, train_dataset):

    if self.items is None:
      self.items = train_dataset.items
    else:
      self.items = np.unique(np.append(self.items, train_dataset.items))

    if self.sequence_ids is None:
      self.sequence_ids = train_dataset.sequence_ids
    else:
      self.sequence_ids = np.unique(np.append(self.sequence_ids, train_dataset.sequence_ids))

    if self.item_based and self.num_items is None:
      self.num_items = int(np.max(self.items)) + 1
    elif self.item_based:
      assert self.num_items >= int(np.max(self.items)) + 1,\
        'The largest item id should be smaller than number of items.' \
        'If your model is not based on items, set item_based to False in Recoder constructor.'

    if self.sequence_based and self.num_sequences is None:
      self.num_sequences = int(np.max(self.sequence_ids)) + 1
    elif self.sequence_based:
      assert self.num_sequences >= int(np.max(self.sequence_ids)) + 1,\
        'The largest sequence id should be smaller than number of sequences.' \
        'If your model is not based on sequences, set sequence_based to False in Recoder constructor.'

  def _state(self):
    return {
      'items': self.items,
      'sequence_ids': self.sequence_ids,
      'num_items': self.num_items,
      'num_sequences': self.num_sequences
    }

  def _init_loaded_state(self, state):
    self.items = state['items']
    self.sequence_ids = state['sequence_ids']
    self.num_items = state['num_items']
    self.num_sequences = state['num_sequences']

  def train(self, train_dataset, val_dataset=None,
            lr=0.001, weight_decay=0, lr_momentum=0.9, num_epochs=1,
            iters_per_epoch=None, batch_size=64, lr_milestones=None,
            lr_gamma=0, negative_sampling=False, max_seq_len=None,
            num_random_neg_samples=0, neg_sampling_alpha=0,
            downsampling_threshold=None, num_data_workers=0,
            model_checkpoint_prefix=None, checkpoint_freq=0,
            eval_freq=0, eval_num_recommendations=None,
            eval_num_sequences=None, metrics=None, eval_batch_size=None,
            eval_input_split=0.5):
    """
    Trains the model

    Args:
      train_dataset (SequenceDataset): train dataset.
      val_dataset (SequenceDataset, optional): validation dataset.
      lr (float, optional): learning rate.
      weight_decay (float, optional): weight decay (L2 normalization).
      num_epochs (int, optional): number of epochs to train the model.
      iters_per_epoch (int, optional): number of training iterations per training epoch. If None,
        one epoch is full number of training samples in the dataset
      batch_size (int, optional): batch size
      max_seq_len (int, optional): maximum sequence length. sequence is randomly tuncated.
      lr_milestones (list, optional): optimizer learning rate epochs milestones (0.1 decay).
      negative_sampling (bool, optional): whether to apply mini-batch based negative sampling or not.
      downsampling_threshold (float, optional): downsampling threshold (same downsampling method used in word2vec).
      num_data_workers (int, optional): number of data workers to use for building the mini-batches.
      checkpoint_freq (int, optional): epochs frequency of saving a checkpoint of the model
      model_checkpoint_prefix (str, optional): model checkpoint save path prefix
      eval_freq (int, optional): epochs frequency of doing an evaluation
      eval_num_recommendations (int, optional): num of recommendations to generate on evaluation
      eval_num_sequences (int, optional): number of sequences from the validation dataset to use for evaluation.
        If None, all sequences in the validation dataset are used for evaluation.
      metrics (list[Metric], optional): list of ``Metric`` used to evaluate the model
      eval_batch_size (int, optional): the size of the evaluation batch
      eval_input_split (float, optional): the split percentage of the input to pass to the model,
        and the remaining split as the target recommendations to evaluate on.
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

    train_dataloader = SequenceDataLoader(train_dataset,
                                          batch_size=batch_size,
                                          negative_sampling=negative_sampling,
                                          downsampling_threshold=downsampling_threshold,
                                          num_random_neg_samples=num_random_neg_samples,
                                          neg_sampling_alpha=neg_sampling_alpha,
                                          num_workers=num_data_workers,
                                          max_seq_len=max_seq_len)
    if val_dataset is not None:
      val_dataloader = SequenceDataLoader(val_dataset,
                                          batch_size=batch_size,
                                          num_items=train_dataloader.num_items,
                                          items_count=train_dataloader.items_count,
                                          negative_sampling=negative_sampling,
                                          downsampling_threshold=downsampling_threshold,
                                          num_random_neg_samples=num_random_neg_samples,
                                          neg_sampling_alpha=neg_sampling_alpha,
                                          num_workers=num_data_workers,
                                          max_seq_len=max_seq_len)
    else:
      val_dataloader = None

    self.train_loop(lr=lr,
                    weight_decay=weight_decay,
                    lr_momentum=lr_momentum,
                    train_dataloader=train_dataloader,
                    val_dataloader=val_dataloader,
                    num_epochs=num_epochs,
                    current_epoch=self.current_epoch,
                    lr_milestones=lr_milestones,
                    lr_gamma=lr_gamma,
                    batch_size=eval_batch_size,
                    model_checkpoint_prefix=model_checkpoint_prefix,
                    checkpoint_freq=checkpoint_freq,
                    eval_freq=eval_freq,
                    metrics=metrics,
                    eval_dataset=val_dataloader.dataset if val_dataloader else None,
                    num_recommendations=eval_num_recommendations,
                    iters_per_epoch=iters_per_epoch,
                    num_sequences=eval_num_sequences,
                    input_split=eval_input_split)

  def predict(self, sequences, return_input=False):
    """
    Returns predictions of the next item in the sequence.

    Args:
      sequences (Sequences): A batch of sequences
      return_input (bool, optional): whether to return the dense input batch

    Returns:
      if ``return_input`` is ``True`` a tuple of the predictions and the input batch
      is returned, otherwise only the predictions are returned
    """
    if self.model is None:
      raise Exception('Model not initialized.')

    self.model.eval()

    batch_collator = SequencesCollator(num_items=self.num_items, negative_sampling=False)

    input = batch_collator.collate(sequences)
    sequence_batch = input
    sequence_batch.to(device=self.device)

    model_output = self.model(sequence_batch)

    compressed_indices = torch.tensor([np.repeat(np.arange(len(sequence_batch.sequence_ids)), sequence_batch.sequences_lens),
                                       sequence_batch.indices[1]],
                                      device=self.device,
                                      dtype=torch.float)

    compressed_size = torch.Size([len(sequence_batch.sequence_ids), sequence_batch.size[1]])

    input_tensor_sparse_compressed = torch.sparse_coo_tensor(compressed_indices,
                                                             sequence_batch.values,
                                                             size=compressed_size,
                                                             device=self.device,
                                                             dtype=torch.float)
    input_dense = input_tensor_sparse_compressed.to_dense()

    return model_output.output, input_dense if return_input else model_output.output

  def recommend(self, sequences, num_recommendations):
    """
    Generate list of next item recommendations for each sequence in ``sequences``.

    Args:
      sequences (Sequences): list of sequences.
      num_recommendations (int): number of recommendations to generate.

    Returns:
      list: list of next item recommendations for each sequence in ``sequences``.
    """
    output, input = self.predict(sequences, return_input=True)
    # Set input items output to -inf so that they don't get recommended
    output[input > 0] = - float('inf')

    top_output, top_ind = torch.topk(output, num_recommendations, dim=1, sorted=True)

    recommendations = top_ind.tolist()

    return recommendations

  def evaluate(self, eval_dataset, num_recommendations, metrics, batch_size=1,
               num_sequences=None, input_split=0.5):
    """
    Evaluates the current model given an evaluation dataset.

    Args:
      eval_dataset (SequenceDataset): evaluation dataset
      num_recommendations (int): number of top recommendations to consider.
      metrics (list): list of ``Metric`` to use for evaluation.
      batch_size (int, optional): batch size of computations.
      num_sequences (int, optional): the number of sequences from the dataset to evaluate on. If None,
        evaluate on all sequences.
      input_split (float, optional): the split percentage of the input to pass to the model,
        and the remaining split as the target recommendations to evaluate on.
    """
    if self.model is None:
      raise Exception('Model not initialized')

    self.model.eval()

    evaluator = SequentialRecommenderEvaluator(self, metrics)

    results = evaluator.evaluate(eval_dataset, batch_size=batch_size, num_sequences=num_sequences,
                                 input_split=input_split, num_recommendations=num_recommendations)
    return results
