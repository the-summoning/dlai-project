import numpy as np
import matplotlib.pyplot as plt
import math

from pathlib import Path


def matrioshka_sizes(min: int, max: int):
    return [ min * 2**i for i in range(int(math.log2(max // min)) + 1)]



def compute_metrics(traj: np.ndarray, W_slices: list[int]):
    results = {}
    
    for d in W_slices:        
        delta = np.diff(traj[:, :, :, :d], axis=1)

        # (Samples, Steps-1, Seq_len)
        radial_tokens = np.linalg.norm(delta, ord=2, axis=-1)
        radial_per_sample = np.mean(radial_tokens, axis=-1)   # (Samples, Steps-1)
        radial_mean = np.mean(radial_per_sample, axis=0)       # (Steps-1,)
        radial_std  = np.std(radial_per_sample,  axis=0)

        d_curr = delta[:, 1:]
        d_prev = delta[:, :-1]
        
        dot_product = np.sum(d_curr * d_prev, axis=-1)
        norm_curr = np.linalg.norm(d_curr, ord=2, axis=-1)
        norm_prev = np.linalg.norm(d_prev, ord=2, axis=-1)
        
        # (Samples, Steps-2, Seq_len)
        #angular_tokens = dot_product / (norm_curr * norm_prev + 1e-8)
        angular_tokens = dot_product / np.maximum(norm_curr * norm_prev, 1e-12)
        angular_per_sample = np.mean(angular_tokens, axis=-1)  # (Samples, Steps-2)
        angular_mean = np.mean(angular_per_sample, axis=0)      # (Steps-2,)
        angular_std  = np.std(angular_per_sample,  axis=0)

        abs_acc_tokens = np.linalg.norm(d_curr - d_prev, ord=2, axis=-1)
        # (Samples, Steps-2, Seq_len)
        acc_tokens = abs_acc_tokens / (norm_curr + norm_prev + 1e-8)
        acc_per_sample = np.mean(acc_tokens, axis=-1)          # (Samples, Steps-2)
        acc_mean = np.mean(acc_per_sample, axis=0)             # (Steps-2,)
        acc_std  = np.std(acc_per_sample,  axis=0)

        results[d] = {
            "radial_mean": radial_mean,
            "radial_std": radial_std,
            "angular_mean": angular_mean,
            "angular_std": angular_std,
            "acc_mean": acc_mean,
            "acc_std": acc_std
        }
    
    return results


def trajectory_with_MRL(
    traj: np.ndarray,
    dims_to_analyze: list[int],
    plot_path: Path,
    x_label: str
):
    results = compute_metrics(traj, dims_to_analyze)

    _, axes = plt.subplots(1, 3, figsize=(18, 5))

    cmap = plt.get_cmap("tab10")

    for i, d in enumerate(dims_to_analyze):
        color = cmap(i % cmap.N)

        res = results[d]
        label = f"Dim {d}"


        steps_radial = np.arange(1, len(res["radial_mean"]) + 1)
        steps_ang = np.arange(2, len(res["angular_mean"]) + 2)
        steps_acc = np.arange(2, len(res["acc_mean"]) + 2)

        axes[0].plot(
            steps_radial,
            res["radial_mean"],
            linestyle="-",
            color=color,
            linewidth=1.2,
            label=label,
        )

        axes[0].fill_between(
            steps_radial,
            np.maximum(
                1e-10,
                res["radial_mean"] - res["radial_std"],
            ),
            res["radial_mean"] + res["radial_std"],
            color=color,
            alpha=0.08,
        )

        axes[1].plot(
            steps_ang,
            res["angular_mean"],
            linestyle="-",
            color=color,
            linewidth=1.2,
            label=label,
        )

        axes[1].fill_between(
            steps_ang,
            np.clip(
                res["angular_mean"] - res["angular_std"],
                -1,
                1,
            ),
            np.clip(
                res["angular_mean"] + res["angular_std"],
                -1,
                1,
            ),
            color=color,
            alpha=0.08,
        )

        # ------------------------------------------------------------------
        # 3. ACCELERATION
        # ------------------------------------------------------------------

        axes[2].plot(
            steps_acc,
            res["acc_mean"],
            linestyle="-",
            color=color,
            linewidth=1.2,
            label=label,
        )

        axes[2].fill_between(
            steps_acc,
            np.maximum(
                1e-10,
                res["acc_mean"] - res["acc_std"],
            ),
            res["acc_mean"] + res["acc_std"],
            color=color,
            alpha=0.08,
        )

    axes[0].set_title(r"$\|\Delta^{(k)}\|_2$", fontweight="bold", fontsize=13)
    axes[0].set_xlabel(x_label)
    axes[0].grid(True, linestyle="-", alpha=0.4, which="both")
    axes[0].legend(loc="upper right", framealpha=0.9, fontsize="small")

    axes[1].set_title(r"$\cos\angle$", fontweight="bold", fontsize=13)
    axes[1].set_xlabel(x_label)
    axes[1].grid(True, linestyle="-", alpha=0.4)
    axes[1].legend(loc="lower right", framealpha=0.9, fontsize="small")

    axes[2].set_title(r"$\hat{a}^{(k)}$", fontweight="bold", fontsize=13)
    axes[2].set_xlabel(x_label)
    axes[2].grid(True, linestyle="-", alpha=0.4, which="both")
    axes[2].legend(loc="upper right", framealpha=0.9, fontsize="small")

    plt.tight_layout()

    plot_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(
        plot_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    return results


def both_trajectories(
    z_traj: np.ndarray,
    y_traj: np.ndarray,
    d: int,
    plot_path: Path,
    x_label: str
):
    results_z = compute_metrics(z_traj, [d])
    results_y = compute_metrics(y_traj, [d])

    _, axes = plt.subplots(1, 3, figsize=(18, 5))

    color_z = "#1f77b4"
    color_y = "#ff7f0e"

    res_z = results_z[d]
    res_y = results_y[d]

    label_z = "z"
    label_y = "y"

    steps_radial_z = np.arange(1, len(res_z["radial_mean"]) + 1)
    steps_radial_y = np.linspace(
        1,
        len(res_z["radial_mean"]),
        len(res_y["radial_mean"]),
    )

    steps_ang_z = np.arange(2, len(res_z["angular_mean"]) + 2)
    steps_ang_y = np.linspace(
        2,
        len(res_z["angular_mean"]) + 1,
        len(res_y["angular_mean"]),
    )

    steps_acc_z = np.arange(2, len(res_z["acc_mean"]) + 2)
    steps_acc_y = np.linspace(
        2,
        len(res_z["acc_mean"]) + 1,
        len(res_y["acc_mean"]),
    )

    axes[0].plot(
        steps_radial_z,
        res_z["radial_mean"],
        linestyle="-",
        color=color_z,
        linewidth=1.2,
        label=label_z,
    )

    axes[0].fill_between(
        steps_radial_z,
        np.maximum(
            1e-10,
            res_z["radial_mean"] - res_z["radial_std"],
        ),
        res_z["radial_mean"] + res_z["radial_std"],
        color=color_z,
        alpha=0.08,
    )

    axes[0].plot(
        steps_radial_y,
        res_y["radial_mean"],
        linestyle="-",
        color=color_y,
        linewidth=1.2,
        label=label_y,
    )

    axes[0].fill_between(
        steps_radial_y,
        np.maximum(
            1e-10,
            res_y["radial_mean"] - res_y["radial_std"],
        ),
        res_y["radial_mean"] + res_y["radial_std"],
        color=color_y,
        alpha=0.06,
    )

    axes[1].plot(
        steps_ang_z,
        res_z["angular_mean"],
        linestyle="-",
        color=color_z,
        linewidth=1.2,
        label=label_z,
    )

    axes[1].fill_between(
        steps_ang_z,
        np.clip(
            res_z["angular_mean"] - res_z["angular_std"],
            -1,
            1,
        ),
        np.clip(
            res_z["angular_mean"] + res_z["angular_std"],
            -1,
            1,
        ),
        color=color_z,
        alpha=0.08,
    )

    axes[1].plot(
        steps_ang_y,
        res_y["angular_mean"],
        linestyle="-",
        color=color_y,
        linewidth=1.2,
        label=label_y,
    )

    axes[1].fill_between(
        steps_ang_y,
        np.clip(
            res_y["angular_mean"] - res_y["angular_std"],
            -1,
            1,
        ),
        np.clip(
            res_y["angular_mean"] + res_y["angular_std"],
            -1,
            1,
        ),
        color=color_y,
        alpha=0.06,
    )

    axes[2].plot(
        steps_acc_z,
        res_z["acc_mean"],
        linestyle="-",
        color=color_z,
        linewidth=1.2,
        label=label_z,
    )

    axes[2].fill_between(
        steps_acc_z,
        np.maximum(
            1e-10,
            res_z["acc_mean"] - res_z["acc_std"],
        ),
        res_z["acc_mean"] + res_z["acc_std"],
        color=color_z,
        alpha=0.08,
    )

    axes[2].plot(
        steps_acc_y,
        res_y["acc_mean"],
        linestyle="-",
        color=color_y,
        linewidth=1.2,
        label=label_y,
    )

    axes[2].fill_between(
        steps_acc_y,
        np.maximum(
            1e-10,
            res_y["acc_mean"] - res_y["acc_std"],
        ),
        res_y["acc_mean"] + res_y["acc_std"],
        color=color_y,
        alpha=0.06,
    )

    axes[0].set_title(r"$\|\Delta^{(k)}\|_2$", fontweight="bold", fontsize=13)
    axes[0].set_xlabel(x_label)
    axes[0].grid(True, linestyle="-", alpha=0.4, which="both")
    axes[0].legend(loc="upper right", framealpha=0.9, fontsize="small")

    axes[1].set_title(r"$\cos\angle$", fontweight="bold", fontsize=13)
    axes[1].set_xlabel(x_label)
    axes[1].grid(True, linestyle="-", alpha=0.4)
    axes[1].legend(loc="lower right", framealpha=0.9, fontsize="small")

    axes[2].set_title(r"$\hat{a}^{(k)}$", fontweight="bold", fontsize=13)
    axes[2].set_xlabel(x_label)
    axes[2].grid(True, linestyle="-", alpha=0.4, which="both")
    axes[2].legend(loc="upper right", framealpha=0.9, fontsize="small")

    plt.tight_layout()

    plot_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(
        plot_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    return results_z, results_y


def trajectory(
    traj: np.ndarray,
    plot_path: Path
):
    steps = traj.shape[1]
    sizes = [512]

    results = compute_metrics(traj, sizes)[sizes[0]]

    _, axes = plt.subplots(1, 3, figsize=(18, 5))

    color = "tab:orange"
    label = "y"

    steps_radial = np.arange(1, steps)
    steps_acc_ang = np.arange(2, steps)

    # 1. Radial (L2)
    axes[0].plot(
        steps_radial,
        results["radial_mean"],
        linestyle='-',
        color=color,
        label=label,
        linewidth=1.2
    )
    axes[0].fill_between(
        steps_radial,
        np.maximum(1e-10, results["radial_mean"] - results["radial_std"]),
        results["radial_mean"] + results["radial_std"],
        color=color,
        alpha=0.08
    )

    # 2. Angular (Coseno)
    axes[1].plot(
        steps_acc_ang,
        results["angular_mean"],
        linestyle='-',
        color=color,
        label=label,
        linewidth=1.2
    )
    axes[1].fill_between(
        steps_acc_ang,
        np.clip(results["angular_mean"] - results["angular_std"], -1, 1),
        np.clip(results["angular_mean"] + results["angular_std"], -1, 1),
        color=color,
        alpha=0.08
    )

    # 3. Acceleration
    axes[2].plot(
        steps_acc_ang,
        results["acc_mean"],
        linestyle='-',
        color=color,
        label=label,
        linewidth=1.2
    )
    axes[2].fill_between(
        steps_acc_ang,
        np.maximum(1e-10, results["acc_mean"] - results["acc_std"]),
        results["acc_mean"] + results["acc_std"],
        color=color,
        alpha=0.08
    )

    x_label = r"Step ($N_{\mathrm{sup}}$)"

    axes[0].set_title(r"$\|\Delta^{(k)}\|_2$", fontweight='bold')
    axes[0].set_xlabel(x_label)
    axes[0].grid(True, linestyle='--', alpha=0.4)
    axes[0].legend()

    axes[1].set_title(r"$\cos\angle$", fontweight='bold')
    axes[1].set_xlabel(x_label)
    axes[1].grid(True, linestyle='--', alpha=0.4)
    axes[1].legend()

    axes[2].set_title(r"$\hat{a}^{(k)}$", fontweight='bold')
    axes[2].set_xlabel(x_label)
    axes[2].grid(True, linestyle='--', alpha=0.4)
    axes[2].legend()

    plt.tight_layout()

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    return results


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



if __name__ == "__main__":
    for path, dims_to_analyze, title in [
        ("/home/davide/Documenti/DLAI project/results/maze5x5_noMRL", [16], "Maze 5x5"),
        ("/home/davide/Documenti/DLAI project/results/maze5x5_withMRL", matrioshka_sizes(4, 16), "Maze 5x5 with MRL"),
        ("/home/davide/Documenti/DLAI project/results/maze10x10_noMRL", [32], "Maze 10x10"),
        ("/home/davide/Documenti/DLAI project/results/maze10x10_withMRL", matrioshka_sizes(4, 32), "Maze 10x10 with MRL"),
        ("/home/davide/Documenti/DLAI project/results/sudoku9x9_noMRL", [16], "Sudoku 9x9"),
        ("/home/davide/Documenti/DLAI project/results/sudoku9x9_withMRL", matrioshka_sizes(4, 16), "Sudoku 9x9 with MRL"),
        ("/home/davide/Documenti/DLAI project/results/sudoku9x9_noMRL_attention", [128], "Sudoku 9x9 (attention)"),
        ("/home/davide/Documenti/DLAI project/results/sudoku9x9_withMRL_attention", matrioshka_sizes(16, 128), "Sudoku 9x9 (attention) with MRL"),
    ]:
        path = Path(path)
        data = np.load(path / 'trajectories.npz', allow_pickle=True)

        z_traj = data['z']
        y_traj = data['y']

        idx = np.random.choice(z_traj.shape[0], size=150, replace=False)

        z_traj = z_traj[idx]
        y_traj = y_traj[idx]

        x_label = r"Step ($T \cdot n \cdot N_{\mathrm{sup}}$)"

        if len(dims_to_analyze) > 1:
            # res = trajectory_with_MRL(
            #     z_traj, 
            #     dims_to_analyze, 
            #     Path(f'./analysis/plots/{path.stem}-z_MRL.png'),
            #     x_label, 
            #     title + ' (z)'
            # )
            
            res = trajectory_with_MRL(
                y_traj, 
                dims_to_analyze, 
                Path(f'./analysis/plots/{path.stem}-y_MRL.png'),
                x_label
            )
    
        else:
            res_z, res_y = both_trajectories(
                z_traj,
                y_traj,
                dims_to_analyze[0],
                plot_path=Path(f'./analysis/plots/{path.stem}.png'),
                x_label=x_label
            )


    trajectory(
        traj=load_batch_trajectories(
            Path('/home/davide/Scaricati/AGI test traj/')
        ),
        plot_path=Path(f'./analysis/plots/arc-ag1_v1.png')
    )

    trajectory(
        traj=load_batch_trajectories(
            Path('/home/davide/Scaricati/AGI2 test traj/')
        ),
        plot_path=Path(f'./analysis/plots/arc-ag1_v2.png')
    )

    # generate_plots_for_y(
    #     y_traj=load_batch_trajectories(
    #         Path('/home/davide/Scaricati/Maze30x30 test traj/')
    #     ),
    #     plot_path=Path(f'./analysis/plots/maze30x30.png'),
    #     title="Maze 30x30",
    # )

    print('Finish')