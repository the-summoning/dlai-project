import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import IncrementalPCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
import joblib


def preprocess_batch(batch_path):
    traj = np.load(batch_path, allow_pickle=True)['y_trajectories']
    print(f'Loaded {batch_path}, {traj.shape}')
    traj = np.transpose(traj, (1, 0, 2, 3))
    y_space_batch = np.mean(traj, axis=2)
    return y_space_batch


def fit_incremental_pca(batches_dir, file_pattern, max_components):
    ipca = IncrementalPCA(n_components=max_components)
    file_list = list(batches_dir.glob(file_pattern))
    file_list_sorted = sorted(file_list, key=lambda p: int(p.stem.split('_')[-1]))
    
    for batch_path in file_list_sorted:
        y_space_batch = preprocess_batch(batch_path)
        _, _, hidden_size = y_space_batch.shape
        z_batch_points = y_space_batch.reshape(-1, hidden_size)
        ipca.partial_fit(z_batch_points)
        print(f"IPCA: Effettuato partial_fit sul file {batch_path.name}")
        
    return ipca


def load_and_transform_all(batches_dir, file_pattern, ipca, max_components):
    all_reduced_batches = []
    total_samples = 0
    steps = None

    file_list = list(batches_dir.glob(file_pattern))
    file_list_sorted = sorted(file_list, key=lambda p: int(p.stem.split('_')[-1]))
    
    for batch_path in file_list_sorted:
        y_space_batch = preprocess_batch(batch_path)
        s, steps_curr, hidden_size = y_space_batch.shape
        
        if steps is None:
            steps = steps_curr
            
        total_samples += s
        y_batch_points = y_space_batch.reshape(-1, hidden_size)
        
        reduced_batch = ipca.transform(y_batch_points)
        reduced_batch_3d = reduced_batch.reshape(s, steps, max_components)
        all_reduced_batches.append(reduced_batch_3d)
        
    y_all_reduced_3d = np.concatenate(all_reduced_batches, axis=0)
    return y_all_reduced_3d, total_samples, steps


# Modificata per accettare un output_dir e salvare ogni modello
def run_grid_search(y_all_reduced_3d, samples, max_components, n_comp_values, K_values, train_idx, test_idx, ipca, models_dir):
    test_reduced_last_step = y_all_reduced_3d[test_idx, -1, :]
    y_test_512 = ipca.inverse_transform(test_reduced_last_step)

    mean_l2_grid = np.zeros((len(n_comp_values), len(K_values)))
    mean_cos_grid = np.zeros((len(n_comp_values), len(K_values)))

    best_l2, best_cos = np.inf, np.inf
    best_l2_cfg, best_cos_cfg = None, None

    for i_nc, n_comp in enumerate(n_comp_values):
        y_comp = y_all_reduced_3d[:, :, :n_comp]

        for i_k, K in enumerate(K_values):
            X = y_comp[:, :K, :].reshape(samples, -1)
            y_target_comp = y_comp[:, -1, :]

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, _ = y_target_comp[train_idx], y_target_comp[test_idx]

            reg = Ridge(alpha=4.0)
            reg.fit(X_train, y_train)

            # --- SALVATAGGIO DEL MODELLO CORRENTE ---
            # Salva il modello usando una convenzione sul nome: ridge_K{valore}_ncomp{valore}.joblib
            model_filename = models_dir / f"ridge_K{K}_ncomp{n_comp}.joblib"
            joblib.dump(reg, model_filename)
            # ----------------------------------------

            pred_comp = reg.predict(X_test)

            pred_full_dim = np.zeros((pred_comp.shape[0], max_components))
            pred_full_dim[:, :n_comp] = pred_comp
            y_pred_512 = ipca.inverse_transform(pred_full_dim)

            l2 = np.linalg.norm(y_test_512 - y_pred_512, axis=1)
            dot = np.sum(y_test_512 * y_pred_512, axis=1)
            norm_true = np.linalg.norm(y_test_512, axis=1)
            norm_pred = np.linalg.norm(y_pred_512, axis=1)
            cos = 1.0 - (dot / (norm_true * norm_pred + 1e-8))

            mean_l2, mean_cos = np.mean(l2), np.mean(cos)
            mean_l2_grid[i_nc, i_k] = mean_l2
            mean_cos_grid[i_nc, i_k] = mean_cos

            if mean_l2 < best_l2:
                best_l2, best_l2_cfg = mean_l2, (K, n_comp)
            if mean_cos < best_cos:
                best_cos, best_cos_cfg = mean_cos, (K, n_comp)
                
    return mean_l2_grid, mean_cos_grid, best_l2, best_l2_cfg, best_cos, best_cos_cfg


def plot_and_save_results(cum_var, max_components, n_comp_values, K_values, mean_l2_grid, mean_cos_grid, output_dir):
    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(1, max_components + 1), cum_var)
    plt.xlabel("n_components")
    plt.ylabel("Cumulative explained variance")
    plt.title("Incremental PCA Explained Variance")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "explained_variance.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.imshow(mean_l2_grid, aspect='auto', origin='lower')
    plt.colorbar(label='Mean L2')
    plt.xticks(range(len(K_values)), K_values)
    plt.yticks(range(len(n_comp_values)), n_comp_values)
    plt.xlabel("K")
    plt.ylabel("n_comp")
    plt.title("Mean L2 Distance")
    plt.tight_layout()
    plt.savefig(output_dir / "heatmap_l2.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.imshow(mean_cos_grid, aspect='auto', origin='lower')
    plt.colorbar(label='Mean Cosine Distance')
    plt.xticks(range(len(K_values)), K_values)
    plt.yticks(range(len(n_comp_values)), n_comp_values)
    plt.xlabel("K")
    plt.ylabel("n_comp")
    plt.title("Mean Cosine Distance")
    plt.tight_layout()
    plt.savefig(output_dir / "heatmap_cosine.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    for i, n_comp in enumerate(n_comp_values):
        plt.plot(K_values, mean_l2_grid[i], label=f"n_comp={n_comp}")
    plt.xlabel("K")
    plt.ylabel("Mean L2")
    plt.title("L2 vs K")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "curves_l2.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    for i, n_comp in enumerate(n_comp_values):
        plt.plot(K_values, mean_cos_grid[i], label=f"n_comp={n_comp}")
    plt.xlabel("K")
    plt.ylabel("Mean Cosine Distance")
    plt.title("Cosine Distance vs K")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "curves_cosine.png", dpi=300)
    plt.close()


def main():
    output_dir = Path("./analysis/pca_gridsearch_plots")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Cartella dove salvare i file .joblib dei modelli e della PCA
    models_dir = Path("./analysis/pca_saved_models")
    models_dir.mkdir(exist_ok=True, parents=True)

    batches_dir = Path('/home/davide/Scaricati/AGI train traj')
    file_pattern = 'batch_*.npz'

    n_comp_values = [64, 128, 256]
    max_components = max(n_comp_values)

    ipca = fit_incremental_pca(batches_dir, file_pattern, max_components)

    # --- SALVATAGGIO DELLA PCA ---
    joblib.dump(ipca, models_dir / "incremental_pca.joblib")
    print(f"\nPCA salvata in: {models_dir / 'incremental_pca.joblib'}")
    # ------------------------------

    cum_var = np.cumsum(ipca.explained_variance_ratio_)
    print("\nExplained variance:")
    for n in n_comp_values:
        if n <= max_components:
            print(f"{n:2d} components -> {100*cum_var[n-1]:.2f}%")

    print("\nTrasformazione dei dati in corso...")
    y_all_reduced_3d, samples, steps = load_and_transform_all(
        batches_dir, file_pattern, ipca, max_components
    )
    print(f"Formato ridotto finale: {y_all_reduced_3d.shape}")

    K_values = list(range(1, steps)) # type: ignore
    indices = np.arange(samples)
    train_idx, test_idx = train_test_split(indices, test_size=0.1, random_state=42)

    print("\nAvvio Grid Search e salvataggio modelli...")
    mean_l2_grid, mean_cos_grid, best_l2, best_l2_cfg, best_cos, best_cos_cfg = run_grid_search(
        y_all_reduced_3d, samples, max_components, n_comp_values, K_values, train_idx, test_idx, ipca, models_dir
    )

    print("\nBEST L2")
    print(f"K={best_l2_cfg[0]} n_comp={best_l2_cfg[1]} L2={best_l2:.6f}") # type: ignore

    print("\nBEST ROSINE")
    print(f"K={best_cos_cfg[0]} n_comp={best_cos_cfg[1]} Cos={best_cos:.6f}") # type: ignore

    plot_and_save_results(cum_var, max_components, n_comp_values, K_values, mean_l2_grid, mean_cos_grid, output_dir)
    print(f"\nGrafici salvati in: {output_dir.resolve()}")
    print(f"Modelli salvati in: {models_dir.resolve()}")


if __name__ == "__main__":
    main()