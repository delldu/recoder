import torch
from torch.utils.data import Dataset, DataLoader, BatchSampler, RandomSampler

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.sputils as sputils

import recoder.data.utils as data_utils


def _compute_sequences_indptr(sequences_lens):
  return np.append(np.cumsum(sequences_lens) - sequences_lens, np.sum(sequences_lens))


def _downsample_sequence(sequences_indptr, interactions,
                         items_inds, sample_mask):
  # reorder the sequence by moving downsampled items down the sequence
  # complexity: O(sum(sequences.sequences_lens))
  for s in range(len(sequences_indptr) - 1):
    first_zero_idx = sequences_indptr[s]
    while first_zero_idx < sequences_indptr[s + 1] and sample_mask[first_zero_idx] != 0:
      first_zero_idx += 1

    current_idx = first_zero_idx
    to_move_idx = first_zero_idx
    while current_idx < sequences_indptr[s + 1]:
      if sample_mask[current_idx] != 0:
        interactions[to_move_idx] = interactions[current_idx]
        interactions[current_idx] = 0
        items_inds[to_move_idx] = items_inds[current_idx]
        items_inds[current_idx] = 0
        to_move_idx += 1
      else:
        interactions[current_idx] = 0
        items_inds[current_idx] = 0

      current_idx += 1


class Sequences:
  """
  Holds a set of sequences in a sparse matrix

  Args:
    sequence_ids (np.array): sequences being represented.
    sequences_matrix (scipy.sparse.csr_matrix): the sequences 2-D matrix of shape
      ``sum(sequences_lens) x M``, where ``M`` is the number of items. The item at position
      ``p`` of a sequence ``s`` is a one-hot encoded vector of size ``M`` stored in
      ``sequences_matrix[sum(sequences_lens[:s]) + p]``.
    sequences_lens (np.array): the lengths of the sequences in the matrix
  """

  def __init__(self, sequence_ids, sequences_matrix, sequences_lens):
    self.sequence_ids = sequence_ids
    self.sequences_matrix = sequences_matrix
    self.sequences_lens = sequences_lens


class SequenceDataset(Dataset):
  """
  Represents a :class:`torch.utils.data.Dataset` that iterates through the sequences.

  Indexing this dataset returns a :class:`Sequences`.

  Args:
    sequences_matrix (scipy.sparse.csr_matrix): the sequences 2-D matrix of shape
      ``sum(sequences_lens) x M``, where ``M`` is the number of items. The item at position
      ``p`` of a sequence ``s`` is a one-hot encoded vector of size ``M`` stored in
      ``sequences_matrix[sum(sequences_lens[:s]) + p]``.
    sequences_lens (np.array): the lengths of the sequences in the matrix
  """

  def __init__(self, sequences_matrix, sequences_lens):
    self.sequences_matrix = sequences_matrix  # type: sparse.csr_matrix
    self.sequences_lens = sequences_lens
    self.sequence_ids = np.arange(len(sequences_lens))
    self.items = np.arange(self.sequences_matrix.shape[1])
    self.sequences_indptr = _compute_sequences_indptr(sequences_lens)

    assert np.sum(sequences_lens) == len(sequences_matrix.data) \
           and np.sum(sequences_lens) + 1 == len(sequences_matrix.indptr)

  def __len__(self):
    return len(self.sequence_ids)

  def __getitem__(self, index):
    assert sputils.issequence(index) or sputils.isintlike(index)

    if sputils.isintlike(index):
      index = [index]

    sequence_ids = np.array(index).reshape(-1,)

    _extended_index = np.hstack([np.arange(self.sequences_indptr[i], self.sequences_indptr[i + 1]) for i in index])

    extracted_sparse_matrix = data_utils.index_sparse_matrix(self.sequences_matrix, _extended_index)

    return Sequences(sequence_ids=sequence_ids, sequences_matrix=extracted_sparse_matrix,
                     sequences_lens=self.sequences_lens[index])


class SequenceDataLoader:
  """
  A ``DataLoader`` similar to ``torch.utils.data.DataLoader`` that handles
  :class:`SequenceDataset` and generate batches with negative sampling.

  By default, if no ``collate_fn`` is provided, the :func:`SequencesCollator.collate` will
  be used, and iterating through this dataloader will return a :class:`SequenceSparseBatch` at each
  iteration, otherwise, the output of the ``collate_fn`` is returned.

  Args:
    dataset (SequenceDataset): dataset from which to load the data
    batch_size (int): number of samples per batch
    negative_sampling (bool, optional): whether to apply mini-batch based negative sampling or not.
    num_sampling_sequences (int, optional): number of sequences to consider for mini-batch based negative
      sampling. This is useful for increasing the number of negative samples while keeping the
      batch-size small. If 0, then num_sampling_sequences will be equal to batch_size.
    num_workers (int, optional): how many subprocesses to use for data loading.
    downsampling_threshold (float, optional): downsampling threshold (same downsampling method used in word2vec).
    collate_fn (callable, optional): A function that transforms a :class:`Sequences` into
      a mini-batch.
  """
  def __init__(self, dataset, batch_size, negative_sampling=False,
               num_sampling_sequences=0, num_workers=0,
               downsampling_threshold=None, collate_fn=None):
    self.dataset = dataset  # type: SequenceDataset
    self.num_sampling_sequences = num_sampling_sequences
    self.num_workers = num_workers
    self.batch_size = batch_size
    self.negative_sampling = negative_sampling
    self.downsampling_threshold = downsampling_threshold

    if self.num_sampling_sequences == 0:
      self.num_sampling_sequences = batch_size

    assert self.num_sampling_sequences >= batch_size, \
      'num_sampling_sequences should be at least equal to the batch_size'

    items_count = np.asarray((self.dataset.sequences_matrix > 0).astype(np.int).sum(axis=0)).reshape(-1,)

    self.sequences_collator = SequencesCollator(batch_size=self.batch_size,
                                                negative_sampling=self.negative_sampling,
                                                downsampling_threshold=downsampling_threshold,
                                                items_count=items_count)

    # Wrapping a BatchSampler within a BatchSampler
    # in order to fetch the whole mini-batch at once
    # from the dataset instead of fetching each sample on its own
    batch_sampler = BatchSampler(BatchSampler(RandomSampler(dataset),
                                              batch_size=self.num_sampling_sequences, drop_last=False),
                                 batch_size=1, drop_last=False)

    if collate_fn is None:
      self._collate_fn = self.sequences_collator.collate
      self._use_default_data_generator = True
    else:
      self._collate_fn = collate_fn
      self._use_default_data_generator = False

    self._dataloader = DataLoader(dataset, batch_sampler=batch_sampler,
                                  num_workers=num_workers, collate_fn=self._collate)

  def _default_data_generator(self):
    for batches in self._dataloader:
      for batch in batches:
        yield batch

  def _collate(self, batch):
    return self._collate_fn(batch[0])

  def __iter__(self):
    if self._use_default_data_generator:
      return self._default_data_generator()

    return self._dataloader.__iter__()

  def __len__(self):
    return int(np.ceil(len(self.dataset) / self.sequences_collator.batch_size))


class SequenceSparseBatch:
  """
  Represents a sparse sequence batch

  Args:
    sequence_ids (torch.LongTensor): users that are in the batch
    items (torch.LongTensor): items that are in the batch
    indices (torch.LongTensor): the indices of the interactions in the sparse matrix
    values (torch.LongTensor): the values of the interactions
    size (torch.Size): the size of the sparse interactions matrix
    sequences_lens (np.array): the length of every sequence in the batch
  """
  def __init__(self, sequence_ids, items,
               indices, values, size,
               sequences_lens):
    self.sequence_ids = sequence_ids
    self.items = items
    self.indices = indices
    self.values = values
    self.size = size
    self.sequences_lens = sequences_lens

    self.sparse_tensor = None

  def to(self, device):
    if self.items is not None:
      self.items = self.items.to(device)

    if self.sequence_ids is not None:
      self.sequence_ids = self.sequence_ids.to(device)

    self.sparse_tensor = torch.sparse.FloatTensor(self.indices,
                                                  self.values,
                                                  self.size).to(device=device)


class SequencesCollator:
  """
  Collator of the sequences contained in a :class:`Sequences`. It collates the sequences
  into multiple :class:`SequenceSparseBatch` of size ``batch_size``.

  Args:
    batch_size (int): number of samples per batch
    negative_sampling (bool, optional): whether to apply mini-batch based negative sampling or not.
    downsampling_threshold (float, optional): downsampling threshold (same downsampling method used in word2vec).
  """
  def __init__(self, batch_size, negative_sampling=False,
               downsampling_threshold=None, items_count=None):

    self.batch_size = batch_size
    self.negative_sampling = negative_sampling
    self.downsampling_threshold = downsampling_threshold
    self.items_count = items_count

    self._items_sampling_prob = None
    if downsampling_threshold is not None and items_count is not None:
      self._items_sampling_prob = data_utils.compute_sampling_probabilities(items_count, downsampling_threshold)

  def collate(self, sequences):
    """
    Collates :class:`Sequences` into multiple :class:`SequenceSparseBatch` of size ``batch_size``.

    Args:
      sequences (Sequences): a :class:`Sequences`.

    Returns:
      list[SequenceSparseBatch]: list of batches.
    """
    batch_sequence_ids = sequences.sequence_ids

    seq_matrix_indptr = sequences.sequences_matrix.indptr
    items_inds = sequences.sequences_matrix.indices
    interactions = sequences.sequences_matrix.data

    sequences_indptr = _compute_sequences_indptr(sequences.sequences_lens)

    if self._items_sampling_prob is not None:
      sample_mask = np.random.binomial(1, p=self._items_sampling_prob[items_inds])
      _downsample_sequence(sequences_indptr=sequences_indptr,
                           interactions=interactions,
                           items_inds=items_inds,
                           sample_mask=sample_mask)

    if self.negative_sampling:
      # The positive item ids in the batch
      # This is simply equivalent to only selecting the non-zero columns
      # in the sparse matrix
      batch_items, items_inds = np.unique(items_inds, return_inverse=True)

      vector_dim = len(batch_items)
      batch_items = torch.LongTensor(batch_items)
    else:
      vector_dim = sequences.sequences_matrix.shape[1]
      batch_items = None

    batch_sequence_ids = torch.LongTensor(batch_sequence_ids)
    slices = []
    for offset in range(0, len(batch_sequence_ids), self.batch_size):
      slice_sequences_lens = sequences.sequences_lens[offset: offset + self.batch_size]
      max_seq_len = np.max(slice_sequences_lens)

      slice_batch_sequence_ids = batch_sequence_ids[offset: offset + self.batch_size]

      slice_indptr = seq_matrix_indptr[sequences_indptr[offset]:sequences_indptr[min(offset + self.batch_size,
                                                                                     len(sequences_indptr) - 1)] + 1]
      slice_items_inds = items_inds[slice_indptr[0]:slice_indptr[-1]]
      slice_inter_vals = interactions[slice_indptr[0]:slice_indptr[-1]]

      slice_sequences_inds = np.hstack([np.arange(row, row + slice_sequences_lens[seq])
                                        for seq, row in enumerate(range(0, len(slice_batch_sequence_ids) * max_seq_len,
                                                                        max_seq_len))])

      indices = torch.LongTensor([slice_sequences_inds, slice_items_inds])
      values = torch.FloatTensor(slice_inter_vals)

      slices.append(SequenceSparseBatch(items=batch_items, sequence_ids=slice_batch_sequence_ids,
                                        indices=indices, values=values,
                                        size=torch.Size([len(slice_batch_sequence_ids)
                                                         * max_seq_len, vector_dim]),
                                        sequences_lens=slice_sequences_lens))

    return slices
