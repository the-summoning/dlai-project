import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
# Importazione necessaria per i grafici 3D nelle versioni precedenti di matplotlib
from mpl_toolkits.mplot3d import Axes3D 

# traj_path ='/home/davide/Documenti/DLAI project/results/maze10x10_noMRL/trajectories.npz'
traj_path ='/home/davide/Documenti/DLAI project/results/sudoku9x9_withMRL/trajectories.npz'
data = np.load(traj_path, allow_pickle=True)
y_traj = data['y']

# traj_path ='/home/davide/Documenti/DLAI project/results/sudoku9x9_noMRL//trajectories.npz' # sample idx 4
# data = np.load(traj_path, allow_pickle=True)
# traj = data['y']

# traj_path ='/home/davide/Scaricati/AGI test traj/batch_1.npz'
# data = np.load(traj_path, allow_pickle=True)
# traj = data['y_trajectories']
# traj = np.transpose(traj, (1, 0, 2, 3))

# 1. Media sui token per il centro di massa
z_space = np.mean(y_traj, axis=2)

# 2. Scegliamo un campione specifico
SAMPLE_IDX = 255 #60
single_trajectory = z_space[SAMPLE_IDX] # Shape: (16, 16)

# 3. Applichiamo la PCA per scendere a 3 dimensioni
pca = PCA(n_components=3)
trajectory_3d = pca.fit_transform(single_trajectory) # Shape: (16, 3)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Quanti step ci sono in questo specifico campione?
num_steps = trajectory_3d.shape[0]

# 1. Invece di usare quiver, coloriamo i segmenti della linea con un gradiente
# Questo mostra la direzione del tempo in modo fluido senza alterare la scala degli assi
cmap = plt.get_cmap('viridis')
for i in range(num_steps - 1):
    # Scegliamo il colore del segmento in base al tempo corrente
    color = cmap(i / (num_steps - 1))
    ax.plot(trajectory_3d[i:i+2, 0], trajectory_3d[i:i+2, 1], trajectory_3d[i:i+2, 2], 
            color=color, linestyle='-', linewidth=3, alpha=0.8, zorder=1)

# 2. Scatter plot dei punti sopra la linea per dare consistenza visiva
sc = ax.scatter(trajectory_3d[:, 0], trajectory_3d[:, 1], trajectory_3d[:, 2], 
                c=range(num_steps), cmap='viridis', s=100, edgecolors='black', alpha=1.0, zorder=3)

# 3. FIX CRUCIALE: Forziamo i limiti degli assi SOLO sulla geometria dei dati reali
# Questo ignora qualsiasi rimasuglio o stranezza di scala passata
padding = 0.05 # 5% di margine intorno ai dati
for axis, dim in zip([ax.set_xlim, ax.set_ylim, ax.set_zlim], [0, 1, 2]):
    data_min = trajectory_3d[:, dim].min()
    data_max = trajectory_3d[:, dim].max()
    data_range = data_max - data_min
    # Evita divisioni per zero se i dati sono piatti su un asse
    if data_range == 0: data_range = 0.1 
    axis(data_min - padding * data_range, data_max + padding * data_range)

# Configurazione grafica e scritte
cbar = fig.colorbar(sc, ax=ax, pad=0.1)
cbar.set_label('Passo di Supervisione (Steps)', rotation=270, labelpad=15)

ax.set_title(f'Traiettoria della Mente Latente 3D (Campione {SAMPLE_IDX})', fontweight='bold', fontsize=12)
ax.set_xlabel('PC1', labelpad=10)
ax.set_ylabel('PC2', labelpad=10)
ax.set_zlabel('PC3', labelpad=10)

ax.view_init(elev=25, azim=35)
ax.grid(True, alpha=0.3)

plt.show()