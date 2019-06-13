from recoder.data.sequence import SequenceDataset, SequenceDataLoader, SequencesCollator, _downsample_sequence
from recoder.data.utils import dataframe_to_seq_csr_matrix

import pandas as pd
import numpy as np
import torch

import pytest

from functools import reduce


def generate_dataframe():
  data = pd.DataFrame()
  data['sequence_id'] = np.random.randint(0, 1000, 100)
  data['sequence'] = [np.random.randint(0, 200, np.random.randint(10, 50)) for _ in range(100)]
  data = data.drop_duplicates(['sequence_id']).reset_index(drop=True)
  return data


@pytest.fixture
def input_dataframe():
  return generate_dataframe()


def test_SequenceDataset(input_dataframe):
  interactions_matrix, sequences_lens, item_id_map, sequence_id_map = \
    dataframe_to_seq_csr_matrix(input_dataframe,
                                sequence_id_col='sequence_id',
                                sequence_col='sequence')

  dataset = SequenceDataset(interactions_matrix, sequences_lens)

  assert len(dataset) == len(np.unique(input_dataframe['sequence_id']))

  inverse_sequence_id_map = {matrix_seq_id: seq_id for seq_id, matrix_seq_id in sequence_id_map.items()}

  index = np.random.randint(0, len(dataset))
  sequence = dataset[index]

  assert sequence.sequences_matrix.shape[0] == \
         len(np.hstack(input_dataframe[input_dataframe.sequence_id == inverse_sequence_id_map[index]].sequence.values))

  assert (sequence.sequences_matrix.indices ==
          np.hstack(input_dataframe[input_dataframe.sequence_id == inverse_sequence_id_map[index]]
                    .sequence.map(lambda itemids: list(map(lambda itemid: item_id_map[itemid], itemids))).values)).all()

  # make sure there's only one element per row
  assert sequence.sequences_matrix.shape[0] == len(sequence.sequences_matrix.data)

  index = np.random.randint(0, len(dataset), np.random.randint(2, 3000))
  sequence = dataset[index]

  assert sequence.sequences_matrix.shape[0] == \
         len(np.hstack(reduce(pd.DataFrame.append, map(lambda i: input_dataframe[input_dataframe.sequence_id == inverse_sequence_id_map[i]], index)).sequence.values))

  assert (sequence.sequences_matrix.indices ==
          np.hstack(reduce(pd.DataFrame.append, map(lambda i: input_dataframe[input_dataframe.sequence_id == inverse_sequence_id_map[i]], index))
                    .sequence.map(lambda itemids: list(map(lambda itemid: item_id_map[itemid], itemids))).values)).all()

  # make sure there's only one element per row
  assert sequence.sequences_matrix.shape[0] == len(sequence.sequences_matrix.data)


@pytest.mark.parametrize("batch_size,num_sampling_sequences",
                         [(5, 0),
                          (5, 10)])
def test_SequenceDataLoader(input_dataframe, batch_size, num_sampling_sequences):

  interactions_matrix, sequences_lens, item_id_map, sequence_id_map = \
    dataframe_to_seq_csr_matrix(input_dataframe,
                                sequence_id_col='sequence_id',
                                sequence_col='sequence')

  dataset = SequenceDataset(interactions_matrix, sequences_lens)

  dataloader = SequenceDataLoader(dataset, batch_size=batch_size,
                                  negative_sampling=True,
                                  num_sampling_sequences=num_sampling_sequences)

  for batch_idx, input in enumerate(dataloader, 1):
    input.to(torch.device('cpu'))
    input_dense = input.sparse_tensor.to_dense()

    assert input_dense.size(0) == batch_size * np.max(input.sequences_lens) \
           or (batch_idx == len(dataloader)
               and input_dense.size(0) == (len(dataset) % batch_size) * np.max(input.sequences_lens))

    assert input_dense.size(1) == len(input.items)

    # make sure one element per row
    assert (input_dense.sum(dim=1) <= 1).all() and (input_dense.sum(dim=1) >= 0).all()


@pytest.mark.parametrize("batch_size",
                         [1, 2, 5, 10, 13])
def test_SequencesCollator(input_dataframe, batch_size):
  sequences_matrix, sequences_lens, item_id_map, sequence_id_map = \
    dataframe_to_seq_csr_matrix(input_dataframe,
                                sequence_id_col='sequence_id',
                                sequence_col='sequence')

  dataset = SequenceDataset(sequences_matrix, sequences_lens)

  batch_collator = SequencesCollator(batch_size=batch_size,
                                     negative_sampling=True)

  big_batch = dataset[np.arange(len(dataset))]

  batches = batch_collator.collate(big_batch)

  assert len(batches) == np.ceil(len(dataset) / batch_size)

  current_batch = 0
  for batch in batches:
    batch.to(torch.device('cpu'))
    input_dense = batch.sparse_tensor.to_dense()
    input_dense = input_dense.view(len(batch.sequence_ids), -1, len(batch.items))

    input_items = batch.items

    batch_sequence_ids = big_batch.sequence_ids[current_batch:current_batch+batch_size]
    batch_seq_lens = big_batch.sequences_lens[current_batch:current_batch+batch_size]
    batch_sparse_matrix = dataset[np.arange(current_batch, min(current_batch + batch_size, len(dataset)))].sequences_matrix

    assert ((input_dense > 0).float().sum(dim=[1, 2]).tolist() == batch_seq_lens).all()

    item_idx_map = {item_id: item_idx for item_idx, item_id in enumerate(input_items.tolist())}

    offset = 0
    for seq_idx in range(len(batch_sequence_ids)):
      for item_pos in range(offset, offset + batch_seq_lens[seq_idx]):
        assert batch_sparse_matrix.indices[item_pos] in input_items
        assert input_dense[seq_idx, item_pos - offset, item_idx_map[batch_sparse_matrix.indices[item_pos]]] == \
               batch_sparse_matrix.data[item_pos]

      offset += batch_seq_lens[seq_idx]

    current_batch += batch_size


def test_downsample_sequence():
  sequences_indptr = [0, 4, 6, 15, 20, 21]
  interactions = np.ones(21)
  sample_mask = np.array([1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0])
  items_inds = np.array([49, 16, 23, 64, 64, 14, 95, 3, 56, 61, 26, 84, 58, 45, 63, 38, 85, 97, 15, 94, 74])

  _downsample_sequence(sequences_indptr=sequences_indptr,
                       interactions=interactions,
                       sample_mask=sample_mask,
                       items_inds=items_inds)

  assert (interactions == np.array([1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0])).all()
  assert (items_inds == np.array([49, 64, 0, 0, 0, 0, 95, 3, 61, 26, 84, 45, 63, 0, 0, 38, 97, 15, 94, 0, 0])).all()
