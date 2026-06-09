import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

def preprocess_batch(batch_path):
    traj = np.load(batch_path, allow_pickle=True)['y_trajectories']
    traj = np.transpose(traj, (1, 0, 2, 3))
    y_space_batch = np.mean(traj, axis=2)
    return y_space_batch

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

def plot_and_save_test_results(n_comp_values, K_values, test_l2_grid, test_cos_grid, output_dir):
    """Genera i grafici delle metriche per il Test Set."""
    output_dir.mkdir(exist_ok=True, parents=True)

    # 1. Heatmap L2
    plt.figure(figsize=(10, 6))
    plt.imshow(test_l2_grid, aspect='auto', origin='lower')
    plt.colorbar(label='Test Mean L2')
    plt.xticks(range(len(K_values)), K_values)
    plt.yticks(range(len(n_comp_values)), n_comp_values)
    plt.xlabel("K")
    plt.ylabel("n_comp")
    plt.title("Test Set: Mean L2 Distance")
    plt.tight_layout()
    plt.savefig(output_dir / "test_heatmap_l2.png", dpi=300)
    plt.close()

    # 2. Heatmap Coseno
    plt.figure(figsize=(10, 6))
    plt.imshow(test_cos_grid, aspect='auto', origin='lower')
    plt.colorbar(label='Test Mean Cosine Distance')
    plt.xticks(range(len(K_values)), K_values)
    plt.yticks(range(len(n_comp_values)), n_comp_values)
    plt.xlabel("K")
    plt.ylabel("n_comp")
    plt.title("Test Set: Mean Cosine Distance")
    plt.tight_layout()
    plt.savefig(output_dir / "test_heatmap_cosine.png", dpi=300)
    plt.close()

    # 3. Curve L2 vs K
    plt.figure(figsize=(10, 6))
    for i, n_comp in enumerate(n_comp_values):
        plt.plot(K_values, test_l2_grid[i], label=f"n_comp={n_comp}", marker='o', markersize=4)
    plt.xlabel("K (Input Steps)")
    plt.ylabel("Mean L2")
    plt.title("Test Set: L2 Distance vs K")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "test_curves_l2.png", dpi=300)
    plt.close()

    # 4. Curve Coseno vs K
    plt.figure(figsize=(10, 6))
    for i, n_comp in enumerate(n_comp_values):
        plt.plot(K_values, test_cos_grid[i], label=f"n_comp={n_comp}", marker='o', markersize=4)
    plt.xlabel("K (Input Steps)")
    plt.ylabel("Mean Cosine Distance")
    plt.title("Test Set: Cosine Distance vs K")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "test_curves_cosine.png", dpi=300)
    plt.close()


def main():
    models_dir = Path("/home/davide/Scrivania/dl project/analysis/pca_saved_models")
    plots_output_dir = Path("/home/davide/Scrivania/dl project/analysis/test_inference_plots")
    
    batches_dir = Path('/home/davide/Scaricati/AGI test traj/')
    file_pattern = 'batch_*.npz'

    n_comp_values = [64, 128, 256]
    max_components = max(n_comp_values)

    print("Caricamento del modello IPCA...")
    ipca = joblib.load(models_dir / "incremental_pca.joblib")

    print("Caricamento e trasformazione dati di TEST...")
    test_reduced_3d, n_test_samples, steps = load_and_transform_all(
        batches_dir, file_pattern, ipca, max_components
    )
    print(f"Formato dati di test ridotti: {test_reduced_3d.shape}")

    test_reduced_last_step = test_reduced_3d[:, -1, :]
    y_test_true_512 = ipca.inverse_transform(test_reduced_last_step)

    K_values = list(range(1, steps)) # type: ignore
    test_l2_grid = np.zeros((len(n_comp_values), len(K_values)))
    test_cos_grid = np.zeros((len(n_comp_values), len(K_values)))

    print("\nInizio ciclo di inferenza...")
    print(f"{'n_comp':<8} | {'K':<5} | {'Test L2':<12} | {'Test Cosine':<12}")
    print("-" * 45)
    
    for i_nc, n_comp in enumerate(n_comp_values):
        test_comp = test_reduced_3d[:, :, :n_comp]

        for i_k, K in enumerate(K_values):
            model_path = models_dir / f"ridge_K{K}_ncomp{n_comp}.joblib"
            
            if not model_path.exists():
                test_l2_grid[i_nc, i_k] = np.nan
                test_cos_grid[i_nc, i_k] = np.nan
                continue
                
            reg = joblib.load(model_path)
            X_test = test_comp[:, :K, :].reshape(n_test_samples, -1)
            pred_comp = reg.predict(X_test)

            pred_full_dim = np.zeros((n_test_samples, max_components))
            pred_full_dim[:, :n_comp] = pred_comp

            y_pred_512 = ipca.inverse_transform(pred_full_dim)

            # Calcolo metriche
            l2_dist = np.linalg.norm(y_test_true_512 - y_pred_512, axis=1)
            dot_product = np.sum(y_test_true_512 * y_pred_512, axis=1)
            norm_true = np.linalg.norm(y_test_true_512, axis=1)
            norm_pred = np.linalg.norm(y_pred_512, axis=1)
            cos_dist = 1.0 - (dot_product / (norm_true * norm_pred + 1e-8))

            mean_l2 = np.mean(l2_dist)
            mean_cos = np.mean(cos_dist)

            test_l2_grid[i_nc, i_k] = mean_l2
            test_cos_grid[i_nc, i_k] = mean_cos

            # Stampa a schermo i risultati correnti riga per riga
            print(f"{n_comp:<8} | {K:<5} | {mean_l2:<12.6f} | {mean_cos:<12.6f}")

    print("\nInferenza completata su tutta la griglia.")
    
    # Calcolo dei minimi escludendo eventuali NaN (se qualche modello mancava)
    min_l2_idx = np.unravel_index(np.nanargmin(test_l2_grid), test_l2_grid.shape)
    min_cos_idx = np.unravel_index(np.nanargmin(test_cos_grid), test_cos_grid.shape)
    
    print("\n" + "="*45)
    print("RIASSUNTO MIGLIORI PERFORMANCE DI TEST:")
    print(f"Miglior L2:     K={K_values[min_l2_idx[1]]}, n_comp={n_comp_values[min_l2_idx[0]]} -> L2={test_l2_grid[min_l2_idx]:.6f}")
    print(f"Miglior Cosine: K={K_values[min_cos_idx[1]]}, n_comp={n_comp_values[min_cos_idx[0]]} -> Cos={test_cos_grid[min_cos_idx]:.6f}")
    print("="*45)

    # Genera e salva i grafici di test
    print(f"\nGenerazione grafici in corso...")
    plot_and_save_test_results(n_comp_values, K_values, test_l2_grid, test_cos_grid, plots_output_dir)
    print(f"Grafici di test salvati con successo in: {plots_output_dir.resolve()}")

if __name__ == "__main__":
    main()






# import numpy as np
# import joblib
# import matplotlib.pyplot as plt
# from pathlib import Path

# def load_raw_test_data(batches_dir, file_pattern):
#     """
#     Carica i dati grezzi di test e li unisce in un unico array 3D.
#     Ritorna un array con forma: (numero_campioni, step_totali, 512)
#     """
#     all_batches = []
#     file_list = list(batches_dir.glob(file_pattern))
#     file_list_sorted = sorted(file_list, key=lambda p: int(p.stem.split('_')[-1]))
    
#     for batch_path in file_list_sorted:
#         # Carica il batch
#         traj = np.load(batch_path, allow_pickle=True)['y_trajectories']
#         traj = np.transpose(traj, (1, 0, 2, 3))
#         # Calcola la media spaziale per ottenere la forma (campioni, step, 512)
#         y_space_batch = np.mean(traj, axis=2)
#         all_batches.append(y_space_batch)
        
#     return np.concatenate(all_batches, axis=0)

# def plot_and_save_test_results(n_comp_values, K_values, test_l2_grid, test_cos_grid, output_dir):
#     """Genera i grafici delle metriche per il Test Set."""
#     output_dir.mkdir(exist_ok=True, parents=True)

#     # 1. Heatmap L2
#     plt.figure(figsize=(10, 6))
#     plt.imshow(test_l2_grid, aspect='auto', origin='lower')
#     plt.colorbar(label='Test Mean L2')
#     plt.xticks(range(len(K_values)), K_values)
#     plt.yticks(range(len(n_comp_values)), n_comp_values)
#     plt.xlabel("K (Input Steps)")
#     plt.ylabel("n_comp")
#     plt.title("Test Set Mean L2 Distance")
#     plt.tight_layout()
#     plt.savefig(output_dir / "test_heatmap_l2.png", dpi=300)
#     plt.close()

#     # 2. Heatmap Coseno
#     plt.figure(figsize=(10, 6))
#     plt.imshow(test_cos_grid, aspect='auto', origin='lower')
#     plt.colorbar(label='Test Mean Cosine Distance')
#     plt.xticks(range(len(K_values)), K_values)
#     plt.yticks(range(len(n_comp_values)), n_comp_values)
#     plt.xlabel("K (Input Steps)")
#     plt.ylabel("n_comp")
#     plt.title("Test Set Mean Cosine Distance")
#     plt.tight_layout()
#     plt.savefig(output_dir / "test_heatmap_cosine.png", dpi=300)
#     plt.close()

#     # 3. Curve L2 vs K
#     plt.figure(figsize=(10, 6))
#     for i, n_comp in enumerate(n_comp_values):
#         plt.plot(K_values, test_l2_grid[i], label=f"n_comp={n_comp}", marker='o', markersize=4)
#     plt.xlabel("K (Input Steps)")
#     plt.ylabel("Mean L2")
#     plt.title("L2 Distance vs K")
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig(output_dir / "test_curves_l2.png", dpi=300)
#     plt.close()

#     # 4. Curve Coseno vs K
#     plt.figure(figsize=(10, 6))
#     for i, n_comp in enumerate(n_comp_values):
#         plt.plot(K_values, test_cos_grid[i], label=f"n_comp={n_comp}", marker='o', markersize=4)
#     plt.xlabel("K (Input Steps)")
#     plt.ylabel("Mean Cosine Distance")
#     plt.title("Cosine Distance vs K")
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig(output_dir / "test_curves_cosine.png", dpi=300)
#     plt.close()


# def main():
#     models_dir = Path("/home/davide/Scrivania/dl project/analysis/pca_saved_models")
#     plots_output_dir = Path("/home/davide/Scrivania/dl project/analysis/test_inference_plots")
    
#     batches_dir = Path('/home/davide/Scaricati/OLD AGI TRAJ/AG1 test traj/')
#     file_pattern = 'batch_*.npz'

#     n_comp_values = [16, 32, 48, 64, 128]
#     max_components = max(n_comp_values)

#     print("Caricamento del modello IPCA...")
#     ipca = joblib.load(models_dir / "incremental_pca.joblib")

#     print("Caricamento dati di TEST ORIGINALI (GREZZI)...")
#     # Carica l'array 3D originale: (campioni, step_totali, 512)
#     y_test_original_3d = load_raw_test_data(batches_dir, file_pattern)
#     n_test_samples, total_steps, hidden_size = y_test_original_3d.shape
#     print(f"Formato dati di test originali: {y_test_original_3d.shape}")

#     # ESTRATTO TARGET: L'ultimo step originale VERO (assoluto) per ogni traiettoria
#     y_test_true_absolute_512 = y_test_original_3d[:, -1, :]
#     print(f"Formato target originale assoluto: {y_test_true_absolute_512.shape}")

#     K_values = list(range(1, total_steps))
#     test_l2_grid = np.zeros((len(n_comp_values), len(K_values)))
#     test_cos_grid = np.zeros((len(n_comp_values), len(K_values)))

#     print("\nInizio ciclo di inferenza esplicito e trasparente...")
#     print(f"{'n_comp':<8} | {'K':<5} | {'Test L2':<12} | {'Test Cosine':<12}")
#     print("-" * 45)
    
#     # Invertiamo l'ordine dei cicli: il ciclo su K diventa esterno per gestire la timeline in modo pulito
#     for i_k, K in enumerate(K_values):
        
#         # 1. TRASPARENZA CRONOLOGICA: Isolo rigorosamente solo i primi K timestep originali
#         # Il futuro (da K in poi) in questo momento non esiste per il codice
#         data_up_to_K = y_test_original_3d[:, :K, :] # Forma: (campioni, K, 512)
        
#         # 2. COMPRESSIONE LOCALIZZATA: Applico la PCA solo sui dati fino a K
#         data_flat = data_up_to_K.reshape(-1, hidden_size)
#         compressed_flat = ipca.transform(data_flat)
#         compressed_3d = compressed_flat.reshape(n_test_samples, K, max_components)

#         for i_nc, n_comp in enumerate(n_comp_values):
#             model_path = models_dir / f"ridge_K{K}_ncomp{n_comp}.joblib"
            
#             if not model_path.exists():
#                 test_l2_grid[i_nc, i_k] = np.nan
#                 test_cos_grid[i_nc, i_k] = np.nan
#                 continue
                
#             # Isolo il numero di componenti richieste dal modello corrente
#             X_test = compressed_3d[:, :, :n_comp].reshape(n_test_samples, -1)
            
#             # 3. REGRESSIONE: Predico lo step finale compresso
#             reg = joblib.load(model_path)
#             pred_comp = reg.predict(X_test)

#             # 4. DECOMPRESSIONE: Riempio di zeri fino a max_components e torno a 512 dimensioni
#             pred_full_dim = np.zeros((n_test_samples, max_components))
#             pred_full_dim[:, :n_comp] = pred_comp
#             y_pred_512 = ipca.inverse_transform(pred_full_dim)

#             # 5. METRICHE TRASPARENTI: Confronto la predizione con l'ultimo step originale VERO
#             # Questo calcolo include sia l'errore di regressione che quello di ricostruzione PCA
#             l2_dist = np.linalg.norm(y_test_true_absolute_512 - y_pred_512, axis=1)
            
#             dot_product = np.sum(y_test_true_absolute_512 * y_pred_512, axis=1)
#             norm_true = np.linalg.norm(y_test_true_absolute_512, axis=1)
#             norm_pred = np.linalg.norm(y_pred_512, axis=1)
#             cos_dist = 1.0 - (dot_product / (norm_true * norm_pred + 1e-8))

#             mean_l2 = np.mean(l2_dist)
#             mean_cos = np.mean(cos_dist)

#             test_l2_grid[i_nc, i_k] = mean_l2
#             test_cos_grid[i_nc, i_k] = mean_cos

#             # Stampa a schermo i risultati correnti riga per riga
#             print(f"{n_comp:<8} | {K:<5} | {mean_l2:<12.6f} | {mean_cos:<12.6f}")

#     print("\nInferenza completata su tutta la griglia.")
    
#     # Calcolo dei minimi escludendo eventuali NaN
#     min_l2_idx = np.unravel_index(np.nanargmin(test_l2_grid), test_l2_grid.shape)
#     min_cos_idx = np.unravel_index(np.nanargmin(test_cos_grid), test_cos_grid.shape)
    
#     print("\n" + "="*45)
#     print("RIASSUNTO MIGLIORI PERFORMANCE DI TEST (ERRORE REALISTICO TOTALE):")
#     print(f"Miglior L2:     K={K_values[min_l2_idx[1]]}, n_comp={n_comp_values[min_l2_idx[0]]} -> L2={test_l2_grid[min_l2_idx]:.6f}")
#     print(f"Miglior Coseno: K={K_values[min_cos_idx[1]]}, n_comp={n_comp_values[min_cos_idx[0]]} -> Cos={test_cos_grid[min_cos_idx]:.6f}")
#     print("="*45)

#     # Genera e salva i grafici di test
#     print(f"\nGenerazione grafici in corso...")
#     plot_and_save_test_results(n_comp_values, K_values, test_l2_grid, test_cos_grid, plots_output_dir)
#     print(f"Grafici di test salvati con successo in: {plots_output_dir.resolve()}")

# if __name__ == "__main__":
#     main()