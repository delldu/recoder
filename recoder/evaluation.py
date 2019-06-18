import scipy.sparse as sparse
import numpy as np

from recoder.data.factorization import RecommendationDataLoader
from recoder.data.sequence import SequenceDataLoader, Sequences

from multiprocessing import Process, Queue


class RecommenderEvaluator(object):
  """
  Evaluates a :class:`recoder.model.factorization.Recoder` given a set of :class:`Metric`

  Args:
    model (Recoder): the model to evaluate
    metrics (list): list of metrics used to evaluate the model
  """

  def __init__(self, model, metrics):
    self.model = model
    self.metrics = metrics

  def evaluate(self, eval_dataset, num_recommendations,
               batch_size=1, num_users=None,
               num_workers=0, input_split=0.5):
    """
    Evaluates the model with an evaluation dataset.

    Args:
      eval_dataset (RecommendationDataset): the dataset to use
        in evaluating the model
      num_recommendations (int): number of recommendations to generate
      batch_size (int): the size of the users batch passed to the model
      num_users (int, optional): the number of users from the dataset to evaluate on. If None,
        evaluate on all users
      num_workers (int, optional): the number of workers to use on evaluating
        the recommended items. This is useful if the model runs on GPU, so the
        evaluation can run in parallel.
      input_split (float, optional): the split percentage of the input to use as user history,
        and the remaining split as the user future interactions.
    Returns:
      dict: A dict mapping each metric to the list of the metric values on each
      user in the dataset.
    """
    dataloader = RecommendationDataLoader(eval_dataset, batch_size=batch_size,
                                          collate_fn=lambda _: _)

    results = {}
    for metric in self.metrics:
      results[metric] = []

    if num_workers > 0:
      input_queue = Queue()
      results_queues = [Queue() for _ in range(num_workers)]

      def evaluate(input_queue, results_queue, metrics):
        results = {}
        for metric in self.metrics:
          results[metric.metric_name] = []

        while True:
          x, y = input_queue.get(block=True)

          if x is None:
            break

          for metric in metrics:
            results[metric.metric_name].append(metric.evaluate(x, y))

        results_queue.put(results)

      workers = [Process(target=evaluate, args=(input_queue, results_queues[p_idx], self.metrics))
                 for p_idx in range(num_workers)]

      for worker in workers:
        worker.start()

    processed_num_users = 0
    for input in dataloader:
      target_mask = np.random.binomial(1, p=(1 - input_split), size=input.interactions_matrix.data.shape)

      target_interactions_matrix = sparse.csr_matrix((input.interactions_matrix.data * target_mask,
                                                      input.interactions_matrix.indices,
                                                      input.interactions_matrix.indptr),
                                                     shape=input.interactions_matrix.shape)

      input.interactions_matrix.data = input.interactions_matrix.data * (1 - target_mask)

      recommendations = self.model.recommend(input, num_recommendations=num_recommendations)

      relevant_items = [target_interactions_matrix[i].nonzero()[1] for i in range(len(input.users))]

      for x, y in zip(recommendations, relevant_items):

        if len(x) == 0 or len(y) == 0:
          continue

        if num_workers > 0:
          input_queue.put((x, y))
        else:
          for metric in self.metrics:
            results[metric].append(metric.evaluate(x, y))

      processed_num_users += len(input.users)
      if num_users is not None and processed_num_users >= num_users:
        break

    for _ in range(num_workers):
      input_queue.put((None, None))

    if num_workers > 0:

      for results_queue in results_queues:
        queue_results = results_queue.get()
        for metric in self.metrics:
          results[metric].extend(queue_results[metric.metric_name])

      for worker in workers:
        worker.join()

    return results


class SequentialRecommenderEvaluator(object):
  """
  Evaluates a :class:`recoder.model.sequence.SequenceRecoder` given a set of :class:`Metric`

  Args:
    model (SequenceRecoder): the model to evaluate
    metrics (list): list of metrics used to evaluate the model
  """

  def __init__(self, model, metrics):
    self.model = model
    self.metrics = metrics

  def evaluate(self, eval_dataset, num_recommendations,
               batch_size=1, num_sequences=None,
               input_split=0.5):
    """
    Evaluates the model with an evaluation dataset.

    Args:
      eval_dataset (SequentialDataset): the dataset to use
        in evaluating the model
      num_recommendations (int): number of recommendations to generate
      batch_size (int): the size of the users batch passed to the model
      num_sequences (int, optional): the number of users from the dataset to evaluate on. If None,
        evaluate on all users
      input_split (float, optional): the split percentage of the input to use as user history,
        and the remaining split as the items to predict.
    Returns:
      dict: A dict mapping each metric to the list of the metric values on each
      user in the dataset.
    """
    def split_sequences(sequences: Sequences):
      input_sequences_lens = (sequences.sequences_lens * input_split).astype(np.int).clip(min=1)
      target_sequences_lens = sequences.sequences_lens - input_sequences_lens

      input_sequences = []
      target_sequences = []
      current_pos = 0
      for seq_len, input_seq_len in zip(sequences.sequences_lens, input_sequences_lens):
        input_sequences.append(sequences.sequences[current_pos: current_pos + input_seq_len])
        target_sequences.append(sequences.sequences[current_pos + input_seq_len: current_pos + seq_len])
        current_pos += seq_len

      input_sequences = np.hstack(input_sequences)
      target_sequences = np.hstack(target_sequences)

      input_sequences = Sequences(sequence_ids=sequences.sequence_ids,
                                  sequences_lens=input_sequences_lens,
                                  sequences=input_sequences)

      target_sequences = Sequences(sequence_ids=sequences.sequence_ids,
                                   sequences_lens=target_sequences_lens,
                                   sequences=target_sequences)

      return input_sequences, target_sequences

    dataloader = SequenceDataLoader(eval_dataset, batch_size=batch_size,
                                    collate_fn=split_sequences)

    results = {}
    for metric in self.metrics:
      results[metric] = []

    processed_num_sequences = 0
    for input, target in dataloader:

      recommendations = self.model.recommend(input, num_recommendations=num_recommendations)

      relevant_items = []
      current_pos = 0
      for seq_len in target.sequences_lens:
        relevant_items.append(target.sequences[current_pos: current_pos + seq_len])
        current_pos += seq_len

      for x, y in zip(recommendations, relevant_items):

        if len(x) == 0 or len(y) == 0:
          continue

        for metric in self.metrics:
          results[metric].append(metric.evaluate(x, y))

      processed_num_sequences += len(input.sequence_ids)
      if num_sequences is not None and processed_num_sequences >= num_sequences:
        break

    return results

