import numpy as np

from puzzle_dataset import PuzzleDataset


# traj = np.load('/home/davide/Scrivania/dl project/maze5x5_results_with_MRL/trajectories.npz', allow_pickle=True)

# z_16 = traj['16'].item()['z']
# z_32 = traj['32'].item()['z']


# print(np.max(np.abs(z_16 - z_32)))

inp_path = "/home/davide/Scrivania/dl project/data/maze5x5/test/inputs.npy"
labels_path = "/home/davide/Scrivania/dl project/data/maze5x5/test/labels.npy"
group_indices_path = "/home/davide/Scrivania/dl project/data/maze5x5/test/group_indices.npy"


inputs = np.load(inp_path)
labels = np.load(labels_path)
group_indices = np.load(group_indices_path)

print("n examples:", len(inputs))
print("n groups:", len(group_indices) - 1)

sizes = np.diff(group_indices)

print("min group size:", sizes.min())
print("max group size:", sizes.max())
print("unique sizes:", np.unique(sizes))



val_dataset = PuzzleDataset(
    inputs_path=inp_path,
    labels_path=labels_path,
    group_indices_path=group_indices_path,
    batch_size=8,
    seed=42,
    ignore_label_id=0
)

count = 0

for batch in val_dataset:
    count += batch["inputs"].shape[0]

print("returned:", count)
print("groups:", val_dataset.num_groups)