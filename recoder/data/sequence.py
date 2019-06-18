import torch
from torch.utils.data import Dataset, DataLoader, BatchSampler, RandomSampler

import numpy as np
import scipy.sparse.sputils as sputils

import recoder.data.utils as data_utils


def _compute_sequences_indptr(sequences_lens):
  return np.append(0, np.cumsum(sequences_lens))


def _sample_sequences(sequences, sample_mask):
  sequences_indptr = _compute_sequences_indptr(sequences.sequences_lens)

  sample_mask = sample_mask.astype(np.bool)

  sampled_sequences_lens_with_zeros = np.array([sample_mask[sequences_indptr[seq]:sequences_indptr[seq+1]].sum()
                                                for seq in range(len(sequences_indptr) - 1)])
  sampled_sequences_mask = sampled_sequences_lens_with_zeros > 0

  sampled_sequence_ids = sequences.sequence_ids[sampled_sequences_mask]
  sampled_sequences_lens = sampled_sequences_lens_with_zeros[sampled_sequences_mask]

  sampled_sequences = sequences.sequences[sample_mask]

  return Sequences(sequence_ids=sampled_sequence_ids,
                   sequences_lens=sampled_sequences_lens,
                   sequences=sampled_sequences)


def _truncate_sequences(sequences, max_seq_len):
  truncated_sequences = []
  truncated_sequences_lens = []
  current_pos = 0
  for seq, seq_len in enumerate(sequences.sequences_lens):
    truncated_len = min(seq_len, max_seq_len)
    start_position = np.random.randint(current_pos, current_pos + seq_len - truncated_len + 1)
    truncated_sequences.append(sequences.sequences[start_position: start_position + truncated_len])
    truncated_sequences_lens.append(truncated_len)

    current_pos += seq_len

  sequences = Sequences(sequence_ids=sequences.sequence_ids,
                        sequences=np.hstack(truncated_sequences),
                        sequences_lens=np.hstack(truncated_sequences_lens))

  return sequences


class Sequences:
  """
  Holds a set of sequences

  Args:
    sequence_ids (np.array): sequences being represented.
    sequences (np.array): the sequences 1-D of shape ``sum(sequences_lens)``.
    sequences_lens (np.array): the lengths of the sequences
  """

  def __init__(self, sequence_ids, sequences, sequences_lens):
    self.sequence_ids = sequence_ids
    self.sequences = sequences
    self.sequences_lens = sequences_lens


class SequenceDataset(Dataset):
  """
  Represents a :class:`torch.utils.data.Dataset` that iterates through the sequences.

  Indexing this dataset returns a :class:`Sequences`.

  Args:
    sequences (np.array): the sequences 1-D of shape ``sum(sequences_lens)``.
    sequences_lens (np.array): the lengths of the sequences, where sequence ``s``
      is contained in ``sequences[sum(sequences_lens[:s+1]):sum(sequences_lens[:s+2])]`` (Note that this
      is optimized internally).
  """

  def __init__(self, sequences, sequences_lens):
    self.sequences = sequences
    self.sequences_lens = sequences_lens
    self.sequence_ids = np.arange(len(sequences_lens))
    self.items = np.unique(self.sequences)
    self.sequences_indptr = _compute_sequences_indptr(sequences_lens)

  def __len__(self):
    return len(self.sequence_ids)

  def __getitem__(self, index):
    assert sputils.isintlike(index) or sputils.issequence(index)

    if sputils.isintlike(index):
      index = [index]

    sequence_ids = np.array(index).reshape(-1,)

    extracted_sequences = np.hstack([self.sequences[self.sequences_indptr[i]: self.sequences_indptr[i + 1]]
                                     for i in index])

    return Sequences(sequence_ids=sequence_ids, sequences=extracted_sequences,
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
    num_workers (int, optional): how many subprocesses to use for data loading.
    downsampling_threshold (float, optional): downsampling threshold (same downsampling method used in word2vec).
    collate_fn (callable, optional): A function that transforms a :class:`Sequences` into
      a mini-batch.
  """
  def __init__(self, dataset, batch_size, negative_sampling=False,
               num_random_neg_samples=0, neg_sampling_alpha=0,
               num_workers=0, downsampling_threshold=None, collate_fn=None,
               items_count=None, num_items=None, max_seq_len=None):
    self.dataset = dataset  # type: SequenceDataset
    self.num_workers = num_workers
    self.batch_size = batch_size
    self.negative_sampling = negative_sampling
    self.downsampling_threshold = downsampling_threshold
    self.num_random_neg_samples = num_random_neg_samples
    self.neg_sampling_alpha = neg_sampling_alpha

    if items_count is None:
      _, items_count = np.unique(self.dataset.sequences, return_counts=True)

    if num_items is None:
      num_items = len(self.dataset.items)

    self.items_count = items_count
    self.num_items = num_items

    self.sequences_collator = SequencesCollator(num_items=num_items,
                                                negative_sampling=self.negative_sampling,
                                                downsampling_threshold=downsampling_threshold,
                                                num_random_neg_samples=num_random_neg_samples,
                                                neg_sampling_alpha=neg_sampling_alpha,
                                                items_count=items_count,
                                                max_seq_len=max_seq_len)

    # Wrapping a BatchSampler within a BatchSampler
    # in order to fetch the whole mini-batch at once
    # from the dataset instead of fetching each sample on its own
    batch_sampler = BatchSampler(BatchSampler(RandomSampler(dataset),
                                              batch_size=self.batch_size, drop_last=False),
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
    for batch in self._dataloader:
      if batch is not None:
        yield batch

  def _collate(self, batch):
    return self._collate_fn(batch[0])

  def __iter__(self):
    if self._use_default_data_generator:
      return self._default_data_generator()

    return self._dataloader.__iter__()

  def __len__(self):
    return int(np.ceil(len(self.dataset) / self.batch_size))


class SequenceSparseBatch:
  """
  Represents a sparse sequence batch

  Args:
    sequence_ids (np.array): sequences that are in the batch
    items (np.array): items that are in the batch
    indices (np.array): the indices of the interactions in the sparse matrix
    values (np.array): the values of the interactions
    size (tuple): the size of the sparse interactions matrix
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
      self.items = torch.tensor(self.items, device=device, dtype=torch.long)

    if self.sequence_ids is not None:
      self.sequence_ids = torch.tensor(self.sequence_ids, device=device, dtype=torch.long)

    self.sparse_tensor = torch.sparse_coo_tensor(self.indices,
                                                 self.values,
                                                 size=self.size,
                                                 device=device,
                                                 dtype=torch.float)


class SequencesCollator:
  """
  Collator of the sequences contained in a :class:`Sequences`. It collates the sequences
  into a :class:`SequenceSparseBatch` of size ``batch_size``.

  Args:
    negative_sampling (bool, optional): whether to apply mini-batch based negative sampling or not.
    downsampling_threshold (float, optional): downsampling threshold (same downsampling method used in word2vec).
  """
  def __init__(self, num_items, negative_sampling=False,
               num_random_neg_samples=0, neg_sampling_alpha=0,
               downsampling_threshold=None, items_count=None,
               max_seq_len=None):

    self.num_items = num_items
    self.negative_sampling = negative_sampling
    self.downsampling_threshold = downsampling_threshold
    self.items_count = items_count
    self.num_random_neg_samples = num_random_neg_samples
    self.neg_sampling_alpha = neg_sampling_alpha
    self.max_seq_len = max_seq_len

    self._items_sampling_prob = None
    if downsampling_threshold is not None and items_count is not None:
      self._items_sampling_prob = data_utils.compute_sampling_probabilities(items_count, downsampling_threshold)

    self._items_neg_sampling_prob = None
    if num_random_neg_samples > 0 and items_count is not None:
      self._items_neg_sampling_prob = np.power(items_count, neg_sampling_alpha)
      self._items_neg_sampling_cumsum = np.append(0, np.cumsum(self._items_neg_sampling_prob))

  def collate(self, sequences):
    """
    Collates :class:`Sequences` into a :class:`SequenceSparseBatch` of size ``batch_size``.

    Args:
      sequences (Sequences): a :class:`Sequences`.

    Returns:
      SequenceSparseBatch: batch.
    """

    if self._items_sampling_prob is not None:
      sample_mask = np.random.binomial(1, p=self._items_sampling_prob[sequences.sequences])
      sequences = _sample_sequences(sequences=sequences, sample_mask=sample_mask)

      if len(sequences.sequence_ids) == 0:
        return None

    if self.max_seq_len is not None:
      sequences = _truncate_sequences(sequences, self.max_seq_len)

    batch_sequence_ids = sequences.sequence_ids

    sequences_items = sequences.sequences

    if self.negative_sampling:
      # The positive item ids in the batch
      # This is simply equivalent to only selecting the non-zero columns
      # in the sparse matrix
      batch_items, sequences_items = np.unique(sequences_items, return_inverse=True)

      if self._items_neg_sampling_prob is not None:
        # np.random.choice is very slow
        # generating random numbers than doing binary search is much faster
        neg_samples = np.searchsorted(self._items_neg_sampling_cumsum,
                                      np.random.rand(self.num_random_neg_samples) * self._items_neg_sampling_cumsum[-1],
                                      side='right') - 1

        # tolerate repetitions and remove duplicates
        neg_samples = np.unique(neg_samples)

        batch_items = np.append(batch_items, neg_samples[~np.in1d(neg_samples, batch_items, assume_unique=True)])

      vector_dim = len(batch_items)
    else:
      vector_dim = self.num_items
      batch_items = None

    max_seq_len = np.max(sequences.sequences_lens)

    rows = np.hstack([np.arange(seq * max_seq_len, seq * max_seq_len + seq_len)
                      for seq, seq_len in enumerate(sequences.sequences_lens)])

    indices = np.array([rows, sequences_items])
    values = np.ones(len(sequences_items))

    return SequenceSparseBatch(items=batch_items, sequence_ids=batch_sequence_ids,
                               indices=indices, values=values,
                               size=(len(batch_sequence_ids) * max_seq_len, vector_dim),
                               sequences_lens=sequences.sequences_lens)
