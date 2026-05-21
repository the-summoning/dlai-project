import numpy as np
import torch

from torch.utils.data import IterableDataset


IGNORE_LABEL_ID = -100


class MazeDataset(IterableDataset):
    """
    Batch semantics:
        - 1 random augmentation per original puzzle
        - no duplicate puzzle inside a batch
    """


    def __init__(
        self,
        inputs_path: str,
        labels_path: str,
        group_indices_path: str,
        batch_size: int,
        ignore_label_id: int,
        seed=42,
    ):
        self.inputs = np.load(inputs_path, mmap_mode="r")
        self.labels = np.load(labels_path, mmap_mode="r")

        # [0, 8, 16, 24, ...]
        self.group_indices = np.load(group_indices_path)

        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)

        self.num_groups = len(self.group_indices) - 1
        self.ignore_label_id = ignore_label_id

    def __iter__(self):
        # shuffle original puzzles
        group_order = self.rng.permutation(self.num_groups)

        batch_inputs = []
        batch_labels = []

        for group_id in group_order:
            # all augmentations belonging to this puzzle
            start = self.group_indices[group_id]
            end = self.group_indices[group_id + 1]

            # choose one random augmentation
            sample_idx = self.rng.integers(start, end)

            batch_inputs.append(self.inputs[sample_idx])
            batch_labels.append(self.labels[sample_idx])

            # emit batch
            if len(batch_inputs) == self.batch_size:
                inputs = torch.tensor(np.stack(batch_inputs), dtype=torch.long)
                labels = torch.tensor(np.stack(batch_labels), dtype=torch.long)

                labels = labels.masked_fill(labels == self.ignore_label_id, IGNORE_LABEL_ID)

                yield {
                    "inputs": inputs,
                    "labels": labels,
                }

                batch_inputs = []
                batch_labels = []