import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


traj_path ='/home/davide/Documenti/DLAI project/maze5x5/val traj no MRL/trajectories.npz'

data = np.load(traj_path, allow_pickle=True)


# 1. Dati: (Samples, Steps, Seq_len, Hidden_size) -> (561, 16, 84, 16)
z_traj = data['z']
z_space = np.mean(z_traj, axis=2)  # Media sui token -> (561, 16, 16)
samples, steps, hidden_size = z_space.shape

# 2. Reshape per fittare la PCA su TUTTI i punti esistenti (561 * 16 = 8976 punti)
# Ogni punto è un vettore di 16 dimensioni
z_all_points = z_space.reshape(-1, hidden_size)  # (8976, 16)

# 3. Fit della PCA globale (puoi impostare n_components=3 per il 3D)
pca = PCA(n_components=2)
points_2d = pca.fit_transform(z_all_points)

# 4. Ricostruiamo la struttura (Samples, Steps, 2)
traj_all_2d = points_2d.reshape(samples, steps, 2)

# 5. Plot dello "Sciame"
plt.figure(figsize=(12, 9))

# Disegniamo prima tutte le linee in grigio chiarissimo e trasparente
for i in range(samples):
    plt.plot(traj_all_2d[i, :, 0], traj_all_2d[i, :, 1], color='gray', alpha=0.03, linewidth=0.5)

# Evidenziamo i punti di inizio (Passo 0) e fine (Passo 15) di tutti i campioni
plt.scatter(traj_all_2d[:, 0, 0], traj_all_2d[:, 0, 1], 
            color='purple', alpha=0.5, s=15, label='Inizio (Passo 0)')
plt.scatter(traj_all_2d[:, -1, 0], traj_all_2d[:, -1, 1], 
            color='gold', alpha=0.7, s=15, label='Fine (Passo 15)')

# Evidenziamo al massimo 3 traiettorie a caso per mostrare il comportamento tipo
for idx in [0, 100, 200]:
    plt.plot(traj_all_2d[idx, :, 0], traj_all_2d[idx, :, 1], linestyle='-', linewidth=2, label=f'Traiettoria campione {idx}')
    plt.scatter(traj_all_2d[idx, :, 0], traj_all_2d[idx, :, 1], c=range(16), cmap='viridis', s=40, zorder=5)

plt.title('Spazio Latente Globale: Lo sciame di traiettorie verso l\'attrattore', fontweight='bold')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.grid(True, alpha=0.2)


plt.show(block=True)

