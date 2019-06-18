import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from recoder.data.sequence import SequenceSparseBatch
from recoder.nn.base import BaseModel, ModelOutput


class SequenceModel(BaseModel):

  def init_model(self, num_items=None, num_sequences=None):
    raise NotImplementedError

  def model_params(self):
    raise NotImplementedError

  def load_model_params(self, model_params):
    raise NotImplementedError

  def forward(self, sequence_batch):
    raise NotImplementedError


class Item2Vec(SequenceModel):

  def __init__(self, embedding_size=None, noise_prob=0,
               sparse=True):
    super().__init__()
    self.embedding_size = embedding_size
    self.noise_prob = noise_prob
    self.sparse = sparse

    self.num_items = None
    self._input_embedding_layer = None
    self._output_embedding_layer = None
    self._noise_layer = None

  def init_model(self, num_items=None, num_sequences=None):
    self.num_items = num_items

    self._input_embedding_layer = nn.Embedding(num_embeddings=num_items, embedding_dim=self.embedding_size,
                                               sparse=self.sparse)
    self._output_embedding_layer = nn.Embedding(num_embeddings=num_items, embedding_dim=self.embedding_size,
                                                sparse=self.sparse)

    self._noise_layer = lambda x: x
    if self.noise_prob > 0:
      self._noise_layer = nn.Dropout(p=self.noise_prob)

    nn.init.xavier_uniform_(self._input_embedding_layer.weight)
    nn.init.xavier_uniform_(self._output_embedding_layer.weight)

  def model_params(self):
    return {'embedding_size': self.embedding_size}

  def load_model_params(self, model_params):
    self.embedding_size = model_params['embedding_size']

  def forward(self, sequence_batch: SequenceSparseBatch):
    input_tensor_sparse = sequence_batch.sparse_tensor

    compressed_indices = torch.tensor([np.repeat(np.arange(len(sequence_batch.sequence_ids)), sequence_batch.sequences_lens),
                                       sequence_batch.indices[1]],
                                      device=input_tensor_sparse.device,
                                      dtype=torch.float)
    compressed_size = torch.Size([len(sequence_batch.sequence_ids), sequence_batch.size[1]])

    input_tensor_sparse_compressed = torch.sparse_coo_tensor(compressed_indices,
                                                             sequence_batch.values,
                                                             size=compressed_size,
                                                             device=input_tensor_sparse.device,
                                                             dtype=torch.float)

    input_tensor = input_tensor_sparse_compressed.to_dense()

    target_tensor = input_tensor

    if sequence_batch.items is not None:
      in_item_embeddings = self._input_embedding_layer(sequence_batch.items)
      out_item_embeddings = self._output_embedding_layer(sequence_batch.items)
    else:
      in_item_embeddings = self._input_embedding_layer.weight
      out_item_embeddings = self._output_embedding_layer.weight

    hidden = torch.matmul(self._noise_layer(F.normalize(input_tensor, p=1, dim=1)), in_item_embeddings)
    output = torch.matmul(hidden, out_item_embeddings.t())

    return ModelOutput(output=output, target=target_tensor)
