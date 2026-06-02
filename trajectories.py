import numpy as np
import matplotlib.pyplot as plt
import math


def profile_latent_geometry(traj, plot_path: str, is_mrl=False, matryoshka_sizes=None):
    """
    Esegue il profiling geometrico dello spazio latente basandosi sui dati di validazione.
    
    Traj: np.ndarray, traiettorie latenti di forma (Samples, Steps, Seq_len, Hidden_size).
    plot_path: str, il percorso dove salvare i plot.
    is_mrl: bool, se True indica che i dati provengono da un modello addestrato con MRL.
    matryoshka_dims: list, le dimensioni delle fette da analizzare (es. [4, 8, 12, 16]).
                     Se None e is_mrl=True, verranno dedotte dall'hidden size.
    """
    # 1. Caricamento dati    
    samples, steps, seq_len, hidden_size = traj.shape
    print(f"Dati caricati. Campioni: {samples}, Step: {steps}, Lunghezza Sequenza: {seq_len}, Hidden Size: {hidden_size}")

    # Configurazione delle fette Matryoshka
    if is_mrl:
        assert matryoshka_sizes is not None
        dims_to_analyze = matryoshka_sizes
    else:
        dims_to_analyze = [hidden_size] # Modello standard analizza solo la dimensione totale

    # Dizionario per salvare i risultati di ogni fetta
    results = {}

    # Analizziamo la geometria per ogni fetta isolata (W_slice)
    for d in dims_to_analyze:
        print(f"Analisi in corso per W_slice = {d}...")
        
        # Estraiamo la sotto-matrice latente corrispondente alla fetta Matryoshka attuale
        z_slice = traj[:, :, :, :d] # (Samples, Steps, Seq_len, d)
        
        # 2. Calcolo di Delta (Variazioni di primo ordine / Velocità)
        # delta ha shape: (Samples, Steps-1, Seq_len, d)
        delta = np.diff(z_slice, axis=1)
        
        # ---- METRICA A: RADIAL REFINEMENT (Norma L2 dei delta, token per token) ----
        # Calcoliamo la norma L2 sull'asse dell'hidden size parziale (axis=-1)
        radial_tokens = np.linalg.norm(delta, ord=2, axis=-1) # (Samples, Steps-1, Seq_len)
        # Reshape per unire campioni e token: (Steps-1, Samples * Seq_len)
        radial_flat = radial_tokens.transpose(1, 0, 2).reshape(steps - 1, -1)
        
        radial_mean = np.mean(radial_flat, axis=1)
        radial_std = np.std(radial_flat, axis=1)
        
        # ---- METRICA B: ANGULAR REFINEMENT (Coseno dell'angolo consecutivo) ----
        d_curr = delta[:, 1:]  # Passi da 1 a steps-2 -> (Samples, Steps-2, Seq_len, d)
        d_prev = delta[:, :-1] # Passi da 0 a steps-3 -> (Samples, Steps-2, Seq_len, d)
        
        # Prodotto scalare token per token sull'asse -1
        dot_product = np.sum(d_curr * d_prev, axis=-1)
        norm_curr = np.linalg.norm(d_curr, ord=2, axis=-1)
        norm_prev = np.linalg.norm(d_prev, ord=2, axis=-1)
        
        angular_tokens = dot_product / (norm_curr * norm_prev + 1e-8) # (Samples, Steps-2, Seq_len)
        angular_flat = angular_tokens.transpose(1, 0, 2).reshape(steps - 2, -1)
        
        angular_mean = np.mean(angular_flat, axis=1)
        angular_std = np.std(angular_flat, axis=1)
        
        # ---- METRICA C: ACCELERAZIONE RELATIVA NORMALIZZATA ----
        # Differenza delle variazioni (Numeratore dell'accelerazione)
        abs_acc_tokens = np.linalg.norm(d_curr - d_prev, ord=2, axis=-1)
        # Denominatore normalizzato (Geometric scaling)
        acc_tokens = abs_acc_tokens / (norm_curr + norm_prev + 1e-8) # (Samples, Steps-2, Seq_len)
        acc_flat = acc_tokens.transpose(1, 0, 2).reshape(steps - 2, -1)
        
        acc_mean = np.mean(acc_flat, axis=1)
        acc_std = np.std(acc_flat, axis=1)
        
        # Salvataggio delle metriche per il plotting
        results[d] = {
            "radial_mean": radial_mean, "radial_std": radial_std,
            "angular_mean": angular_mean, "angular_std": angular_std,
            "acc_mean": acc_mean, "acc_std": acc_std
        }

    # 3. Generazione dei Grafici Diagnostici (Stile Pappone et al.)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    num_slices = len(dims_to_analyze)
    
    # Mappa di colori qualitativa discretizzata per garantire il massimo contrasto
    cmap = plt.cm.get_cmap('tab10', max(10, num_slices))
    colors = [cmap(i) for i in range(num_slices)]

    steps_radial = range(1, steps)      # Da 1 a 15
    steps_acc_ang = range(2, steps)     # Da 2 a 15

    for idx, d in enumerate(dims_to_analyze):
        res = results[d]
        lbl = f"MRL dim: {d}" if is_mrl else "Standard model"
        color = colors[idx]
        
        # Grafico 1: Radial Refinement (Linea continua, senza marcatori)
        axes[0].plot(steps_radial, res["radial_mean"], 
                     linestyle='-', color=color, label=lbl, linewidth=2)
        axes[0].fill_between(steps_radial, np.maximum(1e-10, res["radial_mean"] - res["radial_std"]), 
                             res["radial_mean"] + res["radial_std"], color=color, alpha=0.08)
        
        # Grafico 2: Angular Refinement (Linea continua, senza marcatori)
        axes[1].plot(steps_acc_ang, res["angular_mean"], 
                     linestyle='-', color=color, label=lbl, linewidth=2)
        axes[1].fill_between(steps_acc_ang, np.clip(res["angular_mean"] - res["angular_std"], -1, 1), 
                             np.clip(res["angular_mean"] + res["angular_std"], -1, 1), color=color, alpha=0.08)
        
        # Grafico 3: Accelerazione Relativa (Linea continua, senza marcatori)
        axes[2].plot(steps_acc_ang, res["acc_mean"], 
                     linestyle='-', color=color, label=lbl, linewidth=2)
        axes[2].fill_between(steps_acc_ang, np.maximum(1e-10, res["acc_mean"] - res["acc_std"]), 
                             res["acc_mean"] + res["acc_std"], color=color, alpha=0.08)

    # Configurazione estetica dei subplot
    axes[0].set_title('$\|\Delta^{(k)}\|_2$', fontweight='bold', fontsize=13)
    axes[0].set_xlabel('Step')
    axes[0].grid(True, linestyle='--', alpha=0.4, which="both")
    axes[0].legend(loc='upper right', framealpha=0.9)

    axes[1].set_title('$\cos\\angle$', fontweight='bold', fontsize=13)
    axes[1].set_xlabel('Step')
    axes[1].grid(True, linestyle='--', alpha=0.4)
    axes[1].legend(loc='lower right', framealpha=0.9)

    axes[2].set_title('$\hat{a}^{(k)}$', fontweight='bold', fontsize=13)
    axes[2].set_xlabel('Step')
    axes[2].grid(True, linestyle='--', alpha=0.4, which="both")
    axes[2].legend(loc='upper right', framealpha=0.9)

    plt.suptitle(f"Geometric analysis ({'MRL' if is_mrl else 'without MRL'})", fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    #plt.show()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')

    return results

# --- Esempio di utilizzo dello script ---
if __name__ == "__main__":
    # Sostituisci col percorso reale del tuo file .npz generato dal Val Set
    for latent_type in ['z', 'y']:
        traj_path ='/home/davide/Documenti/DLAI project/maze5x5/val traj no MRL/trajectories.npz'

        data = np.load(traj_path, allow_pickle=True)
        traj = data[latent_type]
        
        print("--- PROFILING MODELLO BASELINE (NO MRL) ---")
        results_no_mrl = profile_latent_geometry(traj, f'maze5x5_no_mrl_{latent_type}.png', is_mrl=False)

        # print('RESULTS NO MRL:')
        # print(results_no_mrl)
        
        ##############################

        traj_path = '/home/davide/Documenti/DLAI project/maze5x5/val traj with MRL/trajectories.npz'

        data = np.load(traj_path, allow_pickle=True)
        traj = data[latent_type]  # Shape attesa: (Samples, Steps, Seq_len, Hidden_size)

        matryoshka_sizes = [
            1 * 2**i for i in range(int(math.log2(16 // 1)) + 1)
        ]
        
        print("--- PROFILING MODELLO con MRL) ---")
        results_no_mrl = profile_latent_geometry(traj, f'maze5x5_with_mrl_{latent_type}.png', is_mrl=True, matryoshka_sizes=matryoshka_sizes)

        # print('RESULTS with MRL:')
        # print(results_no_mrl)