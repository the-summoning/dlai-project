from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def sample_trajectory(traj_path: Path, save_dir: Path, name: str, sample_idx: int):
    data = np.load(traj_path, allow_pickle=True)

    if 'y' in data:
        y_traj = data['y']
    elif 'y_trajectories' in data:
        y_traj = data['y_trajectories']
        y_traj = np.transpose(y_traj, (1, 0, 2, 3))
    else:
        raise KeyError("?")

    y_space = np.mean(y_traj, axis=2)
    single_trajectory_all = y_space[sample_idx]
    single_trajectory = single_trajectory_all[::4]
    num_steps = single_trajectory.shape[0]

    accelerations = []
    deltas = np.diff(single_trajectory, axis=0)

    for k in range(1, len(deltas)):
        delta_cur = deltas[k]
        delta_prev = deltas[k-1]
        diff_delta = delta_cur - delta_prev
        acceleration = np.linalg.norm(diff_delta)
        accelerations.append(acceleration)

    full_accelerations = [np.nan, np.nan] + accelerations

    pca = PCA(n_components=3)
    trajectory_3d = pca.fit_transform(single_trajectory)

    cmap = plt.get_cmap('viridis')
    save_dir.mkdir(parents=True, exist_ok=True)

    # --- Plot 1: 3D Trajectory ---
    fig1 = plt.figure(figsize=(9, 8))
    ax1 = fig1.add_subplot(111, projection='3d')

    for i in range(num_steps - 1):
        color = cmap(i / (num_steps - 1))
        ax1.plot(trajectory_3d[i:i+2, 0], trajectory_3d[i:i+2, 1], trajectory_3d[i:i+2, 2],
                 color=color, linestyle='-', linewidth=3, alpha=0.8, zorder=1)

    sc1 = ax1.scatter(trajectory_3d[:, 0], trajectory_3d[:, 1], trajectory_3d[:, 2], # type: ignore
                      c=range(num_steps), cmap='viridis', s=100, edgecolors='black', alpha=1.0, zorder=3)

    padding = 0.05
    for axis, dim in zip([ax1.set_xlim, ax1.set_ylim, ax1.set_zlim], [0, 1, 2]):
        data_min = trajectory_3d[:, dim].min()
        data_max = trajectory_3d[:, dim].max()
        data_range = data_max - data_min
        if data_range == 0: data_range = 0.1
        axis(data_min - padding * data_range, data_max + padding * data_range)

    #ax1.set_title(f'Trajectory (PCA)', fontweight='bold', fontsize=12)
    ax1.set_xlabel('PC 1', labelpad=10)
    ax1.set_ylabel('PC 2', labelpad=10)
    ax1.set_zlabel('PC 3', labelpad=10)
    ax1.view_init(elev=25, azim=35)
    ax1.grid(True, alpha=0.3)

    cbar1 = fig1.colorbar(sc1, ax=ax1, pad=0.05, shrink=0.7)
    cbar1.set_label('Supervision steps', rotation=270, labelpad=15)

    plt.savefig(save_dir / f"{name}_trajectory.png", dpi=300, bbox_inches='tight')
    plt.close(fig1)

    # --- Plot 2: Acceleration ---
    fig2 = plt.figure(figsize=(9, 8))
    ax2 = fig2.add_subplot(111)
    steps = np.arange(num_steps)

    for i in range(2, num_steps - 1):
        color = cmap(i / (num_steps - 1))
        ax2.plot(steps[i:i+2], full_accelerations[i:i+2], color=color, linestyle='-', linewidth=3, alpha=0.8)

    sc2 = ax2.scatter(steps[2:], full_accelerations[2:], c=range(2, num_steps), cmap='viridis',
                      s=100, edgecolors='black', zorder=3)

    #ax2.set_title('Unnormalized Acceleration over time', fontweight='bold', fontsize=12)
    ax2.set_xlabel('Supervision steps', fontsize=11)
    ax2.set_ylabel('Acceleration', fontsize=11)
    ax2.set_xlim(-0.5, num_steps - 0.5)
    ax2.set_xticks(range(num_steps))
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_ylim(bottom=-0.05)

    cbar2 = fig2.colorbar(sc2, ax=ax2, pad=0.05, shrink=0.7)
    cbar2.set_label('Supervision steps', rotation=270, labelpad=15)

    plt.savefig(save_dir / f"{name}_acceleration.png", dpi=300, bbox_inches='tight')
    plt.close(fig2)


if __name__ == '__main__':
    out_dir = Path("./analysis/sample-trajectories")
    out_dir.mkdir(parents=True, exist_ok=True)

    for traj_path, sample_idx, name in [
        ('/home/davide/Documenti/DLAI project/results/maze10x10_noMRL/trajectories.npz', 178, "maze10x10"),
        ('/home/davide/Documenti/DLAI project/results/sudoku9x9_noMRL/trajectories.npz', 60, "sudoku9x9"),
        ('/home/davide/Documenti/DLAI project/results/sudoku9x9_noMRL_attention/trajectories.npz', 90, "sudoku9x9-attn"),
        ('/home/davide/Documenti/DLAI project/results/maze5x5_noMRL/trajectories.npz', 3, "maze5x5"),
    ]:
        sample_trajectory(Path(traj_path), out_dir, name, sample_idx)