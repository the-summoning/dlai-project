import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pathlib import Path

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


traj = load_batch_trajectories(Path('/home/davide/Scaricati/AGI2 train traj'))


# traj = np.load('/home/davide/Documenti/DLAI project/results/maze5x5_withMRL/trajectories.npz', allow_pickle=True)['y']
idx = np.random.choice(traj.shape[0], size=250, replace=False)
traj = traj[idx]


y_space = np.mean(traj, axis=2)  # (N_samples, steps, hidden_size)
samples, steps, hidden_size = y_space.shape

# PCA globale a 3 componenti
z_all_points = y_space.reshape(-1, hidden_size)
pca_3d = PCA(n_components=3)
points_3d = pca_3d.fit_transform(z_all_points)
traj_all_3d = points_3d.reshape(samples, steps, 3)

# Colormap: viola→giallo lungo gli step
cmap = plt.cm.plasma
colors = cmap(np.linspace(0, 1, steps))  # (steps, 4) RGBA

fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# Linee colorate per step
for i in range(samples):
    for t in range(steps - 1):
        ax.plot(
            traj_all_3d[i, t:t+2, 0],
            traj_all_3d[i, t:t+2, 1],
            traj_all_3d[i, t:t+2, 2],
            color=colors[t],
            alpha=0.06,
            linewidth=0.8
        )

# Punti inizio e fine
ax.scatter(traj_all_3d[:, 0, 0],  traj_all_3d[:, 0, 1],  traj_all_3d[:, 0, 2],
           color=colors[0],  alpha=0.5, s=12, label='Passo 0 (inizio)', zorder=5)
ax.scatter(traj_all_3d[:, -1, 0], traj_all_3d[:, -1, 1], traj_all_3d[:, -1, 2],
           color=colors[-1], alpha=0.7, s=12, label=f'Passo {steps-1} (fine)', zorder=5)

# Colorbar manuale per la progressione temporale
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=steps-1))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, pad=0.1, shrink=0.6)
cbar.set_label('Supervision step', fontweight='bold')

ax.set_title('Spazio Latente Globale 3D: Lo sciame verso l\'attrattore', fontweight='bold')
ax.set_xlabel('PC1', fontweight='bold')
ax.set_ylabel('PC2', fontweight='bold')
ax.set_zlabel('PC3', fontweight='bold')
ax.legend(loc='upper left', fontsize=8)
ax.view_init(elev=20, azim=45)

plt.tight_layout()
plt.savefig('latent_trajectory_3d.png', dpi=150, bbox_inches='tight')
plt.show()