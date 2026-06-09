from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def load_batch_trajectories(batches_dir: Path, first_n_batches: int | None = None):
    file_list = list(batches_dir.glob('batch_*.npz'))
    file_list_sorted = sorted(file_list, key=lambda p: int(p.stem.split('_')[-1]))

    if first_n_batches is not None:
        file_list_sorted = file_list_sorted[:first_n_batches]

    all_y_batches = []
    
    for batch_path in file_list_sorted:
        data = np.load(batch_path, allow_pickle=True)
        y_batch = data['y_trajectories']
        print(f"Loaded {batch_path.name} with shape {y_batch.shape}")

        y_batch = np.transpose(y_batch, (1, 0, 2, 3))
        
        all_y_batches.append(y_batch)

    y_all_trajectories = np.concatenate(all_y_batches, axis=0)

    del all_y_batches
    
    print(f"Final trajectories shape: {y_all_trajectories.shape}")

    return y_all_trajectories



#traj = np.load('/home/davide/Documenti/DLAI project/maze5x5_withMRL/trajectories.npz', allow_pickle=True)['y']

traj = load_batch_trajectories(Path('/home/davide/Scaricati/Maze30x30 train traj/'))

space = np.mean(traj, axis=2)  # (samples, steps, hidden)

samples, steps, hidden_size = space.shape

# 2. Reshape globale
z_all_points = space.reshape(-1, hidden_size)

# 3. PCA globale a 2 componenti
pca_2d = PCA(n_components=2)
points_2d = pca_2d.fit_transform(z_all_points)

# 4. Ricostruzione traiettorie
traj_all_2d = points_2d.reshape(samples, steps, 2)

# 5. Plot
fig, ax = plt.subplots(figsize=(10, 8))

# Sciame di traiettorie
for i in range(samples):
    ax.plot(
        traj_all_2d[i, :, 0],
        traj_all_2d[i, :, 1],
        color="gray",
        alpha=0.08,
        linewidth=1,
    )

# Punti iniziali
ax.scatter(
    traj_all_2d[:, 0, 0],
    traj_all_2d[:, 0, 1],
    color="purple",
    alpha=0.4,
    s=15,
    label="Inizio (Passo 0)",
)

# Punti finali
ax.scatter(
    traj_all_2d[:, -1, 0],
    traj_all_2d[:, -1, 1],
    color="gold",
    alpha=0.6,
    s=15,
    label=f"Fine (Passo {steps-1})",
)

ax.set_title(
    "Spazio Latente Globale 2D: Lo sciame verso l'attrattore",
    fontweight="bold",
)

ax.set_xlabel("PC1", fontweight="bold")
ax.set_ylabel("PC2", fontweight="bold")

ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()