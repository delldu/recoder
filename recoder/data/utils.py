import scipy.sparse as sparse
import scipy.sparse.sputils as sputils
from scipy.sparse import coo_matrix
import numpy as np

import time


_CSR_MATRIX_INDEX_SIZE_LIMIT = 2000


def index_sparse_matrix(sparse_matrix, index):

  if sputils.issequence(index) and len(index) > _CSR_MATRIX_INDEX_SIZE_LIMIT:
    # It happens that scipy implements the indexing of a csr_matrix with a list using
    # matrix multiplication, which gets to be an issue if the size of the index list is
    # large and lead to memory issues
    # Reference: https://stackoverflow.com/questions/46034212/sparse-matrix-slicing-memory-error/46040827#46040827

    # In order to solve this issue, simply chunk the index into smaller indices of
    # size CSR_MATRIX_INDEX_SIZE_LIMIT and then stack the extracted chunks

    sparse_matrix_slices = []
    for offset in range(0, len(index), _CSR_MATRIX_INDEX_SIZE_LIMIT):
      sparse_matrix_slices.append(sparse_matrix[index[offset: offset + _CSR_MATRIX_INDEX_SIZE_LIMIT]])

    extracted_sparse_matrix = sparse.vstack(sparse_matrix_slices)
  else:
    extracted_sparse_matrix = sparse_matrix[index]

  return extracted_sparse_matrix


def compute_sampling_probabilities(items_count, downsampling_threshold):
  # based on word2vec downsampling method
  count_threshold = downsampling_threshold * np.sum(items_count)
  items_sampling_prob = (np.sqrt(items_count / count_threshold) + 1) * (count_threshold / items_count)
  items_sampling_prob = np.clip(items_sampling_prob, 0, 1)

  return items_sampling_prob


def dataframe_to_csr_matrix(dataframe, user_col, item_col,
                            inter_col=None, item_id_map=None,
                            user_id_map=None):
  """
  Converts a :class:`pandas.DataFrame` of users and items interactions into a :class:`scipy.sparse.csr_matrix`.

  This function returns a tuple of the interactions sparse matrix, a `dict` that maps
  from original item ids in the dataframe to the 0-based item ids, and similarly a `dict` that maps
  from original user ids in the dataframe to the 0-based user ids.

  Args:
    dataframe (pandas.DataFrame): A dataframe containing users and items interactions
    user_col (str): users column name
    item_col (str): items column name
    inter_col (str): user-item interaction value column name
    item_id_map (dict, optional): A dictionary mapping from original item ids into 0-based item ids. If not given,
      the map will be generated using the items column in the dataframe
    user_id_map (dict, optional): A dictionary mapping from original user ids into 0-based user ids. If not given,
      the map will be generated using the users column in the dataframe

  Returns:
    tuple: A tuple of the `csr_matrix`, a :class:`dict` `item_id_map`, and a :class:`dict` `user_id_map`
  """

  if user_id_map is None:
    users = dataframe[user_col].unique()
    user_id_map = {user: userid for userid, user in enumerate(users)}

  if item_id_map is None:
    items = dataframe[item_col].unique()
    item_id_map = {item: itemid for itemid, item in enumerate(items)}

  matrix_size = (len(user_id_map.keys()), len(item_id_map.keys()))

  matrix_users = dataframe[user_col].map(user_id_map)
  matrix_items = dataframe[item_col].map(item_id_map)
  if inter_col is not None:
    matrix_inters = dataframe[inter_col]
  else:
    matrix_inters = np.ones(len(matrix_users))

  csr_matrix = coo_matrix((matrix_inters, (matrix_users, matrix_items)), shape=matrix_size).tocsr()

  return csr_matrix, item_id_map, user_id_map


def dataframe_to_seq_csr_matrix(dataframe, sequence_id_col, sequence_col,
                                item_id_map=None, sequence_id_map=None):
  """
  Converts a :class:`pandas.DataFrame` of users and items interactions into a :class:`scipy.sparse.csr_matrix`.

  This function returns a tuple of the interactions sparse matrix, a `dict` that maps
  from original item ids in the dataframe to the 0-based item ids, and similarly a `dict` that maps
  from original user ids in the dataframe to the 0-based user ids.

  Args:
    dataframe (pandas.DataFrame): A dataframe containing users and items interactions
    sequence_id_col (str): sequence ids column name
    sequence_col (str): sequence column name
    item_id_map (dict, optional): A dictionary mapping from original item ids into 0-based item ids. If not given,
      the map will be generated using the items column in the dataframe
    sequence_id_map (dict, optional): A dictionary mapping from original sequence ids into 0-based sequence ids.
      If not given, the map will be generated using the sequence_id column in the dataframe

  Returns:
    tuple: A tuple of the `csr_matrix`, a :class:`dict` `item_id_map`, and a :class:`dict` `user_id_map`
  """

  if sequence_id_map is None:
    sequences = dataframe[sequence_id_col].unique()
    sequence_id_map = {sequence: sequence_id for sequence_id, sequence in enumerate(sequences)}

  if item_id_map is None:
    items = np.unique(np.hstack(dataframe[sequence_col]))
    item_id_map = {item: itemid for itemid, item in enumerate(items)}

  dataframe[sequence_id_col + '_matrix'] = dataframe[sequence_id_col].map(sequence_id_map)
  dataframe.sort_values(inplace=True, by=sequence_id_col + '_matrix')

  sequences_lens = dataframe[sequence_col].map(len).values

  matrix_rows = np.arange(0, np.sum(sequences_lens))
  matrix_items = np.hstack(dataframe[sequence_col].map(lambda seq: list(map(lambda itemid: item_id_map[itemid], seq))))
  matrix_values = np.ones(len(matrix_rows))

  matrix_size = (len(matrix_rows), len(item_id_map.keys()))

  csr_matrix = coo_matrix((matrix_values, (matrix_rows, matrix_items)), shape=matrix_size).tocsr()

  return csr_matrix, sequences_lens, item_id_map, sequence_id_map
