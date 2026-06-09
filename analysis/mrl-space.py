import numpy as np

# =========================
# LOAD + FLATTEN
# =========================

traj_path = '/home/davide/Documenti/DLAI project/maze10x10_withMRL/trajectories.npz'
data = np.load(traj_path, allow_pickle=True)
z_traj = data['y']

n_samples, n_timesteps, seq_len, hidden_size = z_traj.shape

X_full = z_traj.reshape(-1, hidden_size)
slices = [2, 4, 8, 12, 16]

# Centralizziamo subito l'intero dataset per garantire la correttezza geometrica
X_full_centered = X_full - X_full.mean(axis=0)

# Creiamo le slice direttamente dai dati già centralizzati
X_slices_centered = {d: X_full_centered[:, :d] for d in slices}

# =========================
# (A) PROJECTION RESIDUAL ANALYSIS (FIXED MEMORY)
# =========================

print("\n=== PROJECTION RESIDUAL ANALYSIS ===")

def projection_residual(X_high, X_low):
    """
    Proietta X_high sul sottospazio lineare generato da X_low 
    e misura l'errore residuo di Frobenius normalizzato.
    """
    X_low_pinv = np.linalg.pinv(X_low) # Shape: (d, N)
    
    # IMPORTANTE: Le parentesi evitano l'allocazione di una matrice N x N
    X_proj = X_low @ (X_low_pinv @ X_high) # Shape: (N, d) @ (d, D) -> (N, D)

    # Errore residuo relativo (Norma di Frobenius)
    residual = np.linalg.norm(X_high - X_proj, ord='fro') / np.linalg.norm(X_high, ord='fro')
    return residual

for d in slices[:-1]:
    res = projection_residual(X_full_centered, X_slices_centered[d])
    print(f"Slice d={d:2d} vs full -> residual error: {res:.4f}")


# =========================
# (B) EXPLAINED VARIANCE TRANSFER (FIXED MEMORY)
# =========================

print("\n=== EXPLAINED VARIANCE TRANSFER ===")

def explained_variance_transfer(X_low, X_high):
    """
    Quanta della varianza totale dello spazio X_high è catturata 
    (spiegata) dal sottospazio lineare di X_low.
    """
    X_low_pinv = np.linalg.pinv(X_low)
    
    # Anche qui, parentesi per calcolare prima lo spazio latente ridotto
    X_proj = X_low @ (X_low_pinv @ X_high)

    # Varianza totale (somma dei quadrati dei dati centralizzati)
    var_total = np.sum(X_high ** 2)
    # Varianza spiegata dalla proiezione
    var_proj = np.sum(X_proj ** 2)

    return var_proj / var_total

for d in slices[:-1]:
    ev = explained_variance_transfer(X_slices_centered[d], X_full_centered)
    print(f"Slice d={d:2d} explains full variance -> {ev*100:.2f}%")