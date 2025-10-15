import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
import pandas as pd
import os

M, N = 60, 60            # tamaño del grid
I0 = 15                  # infectados iniciales
beta = 0.3               # tasa de infección
gamma = 0.05             # tasa de recuperación
r = 1                    # radio de vecindad (en celdas)
Tmax = 150               # pasos totales de simulación

target_duration_sec = 15.0
fps = 10
dpi = 100
out_path_gif = "sir_cellular.gif"
out_csv = "sir_cellular_timeseries.csv"

total_steps = Tmax
n_frames = int(target_duration_sec * fps)
steps_per_frame = max(1, total_steps // n_frames)
n_frames = total_steps // steps_per_frame
print(f"Total steps: {total_steps}, Frames: {n_frames}, Steps/frame: {steps_per_frame}")

STATE_S, STATE_I, STATE_R = 0, 1, 2
cmap = ListedColormap(["blue", "red", "green"])  # colores consistentes SIR

rng = np.random.default_rng(seed=42)
grid = np.zeros((M, N), dtype=int)
infected_idx = rng.choice(M * N, size=I0, replace=False)
for idx in infected_idx:
    grid[idx // N, idx % N] = STATE_I

S_hist, I_hist, R_hist, time_hist = [], [], [], []
frames_grid, frames_time = [], []

def count_infected_neighbors(g, x, y):
    count = 0
    for dx in range(-r, r + 1):
        for dy in range(-r, r + 1):
            if dx == 0 and dy == 0:
                continue
            nx, ny = (x + dx) % M, (y + dy) % N
            if g[nx, ny] == STATE_I:
                count += 1
    return count

def step(grid):
    new_grid = grid.copy()
    for i in range(M):
        for j in range(N):
            if grid[i, j] == STATE_S:
                infected_neighbors = count_infected_neighbors(grid, i, j)
                if np.random.rand() < 1 - (1 - beta) ** infected_neighbors:
                    new_grid[i, j] = STATE_I
            elif grid[i, j] == STATE_I:
                if np.random.rand() < gamma:
                    new_grid[i, j] = STATE_R
    return new_grid


print("Ejecutando simulación...")
t = 0
for step_idx in range(total_steps):
    if step_idx % steps_per_frame == 0 or step_idx == total_steps - 1:
        frames_grid.append(grid.copy())
        frames_time.append(t)
        S_hist.append(np.sum(grid == STATE_S))
        I_hist.append(np.sum(grid == STATE_I))
        R_hist.append(np.sum(grid == STATE_R))
        time_hist.append(t)
    grid = step(grid)
    t += 1

print("Simulación finalizada. Frames generados:", len(frames_grid))


fig = plt.figure(figsize=(12, 6), dpi=dpi)
ax_grid = fig.add_axes([0.01, 0.05, 0.48, 0.9])  # panel izquierdo
ax_plot = fig.add_axes([0.53, 0.05, 0.46, 0.9])  # panel derecho

im = ax_grid.imshow(frames_grid[0], cmap=cmap, vmin=0, vmax=2)
ax_grid.set_title("Autómata Celular — Modelo SIR")
cbar = plt.colorbar(im, ax=ax_grid, ticks=[0, 1, 2])
cbar.ax.set_yticklabels(['S', 'I', 'R'])
legend_elements = [
    Line2D([0], [0], marker='s', color='w', label='Susceptible (S)', markerfacecolor='blue', markersize=10),
    Line2D([0], [0], marker='s', color='w', label='Infectado (I)', markerfacecolor='red', markersize=10),
    Line2D([0], [0], marker='s', color='w', label='Recuperado (R)', markerfacecolor='green', markersize=10)
]
ax_grid.legend(handles=legend_elements, loc='upper right')
time_text = ax_grid.text(0.02, 1.02, f"t = {frames_time[0]:.1f}",
                         transform=ax_grid.transAxes, fontsize=10, va='bottom')

# --- Panel de curvas SIR ---
ax_plot.set_xlim(0, Tmax)
ax_plot.set_ylim(0, M * N)
ax_plot.set_xlabel("Tiempo (pasos)")
ax_plot.set_ylabel("Número de celdas")
ax_plot.set_title("Curvas de evolución S(t), I(t), R(t)")
line_S, = ax_plot.plot([], [], label='Susceptibles (S)', color='blue', lw=2)
line_I, = ax_plot.plot([], [], label='Infectados (I)', color='red', lw=2)
line_R, = ax_plot.plot([], [], label='Recuperados (R)', color='green', lw=2)
ax_plot.legend(loc='upper right')


def update_frame(frame_idx):
    im.set_data(frames_grid[frame_idx])
    time_text.set_text(f"t = {frames_time[frame_idx]:.1f}")
    times = np.array(time_hist[:frame_idx + 1])
    Svals = np.array(S_hist[:frame_idx + 1])
    Ivals = np.array(I_hist[:frame_idx + 1])
    Rvals = np.array(R_hist[:frame_idx + 1])
    line_S.set_data(times, Svals)
    line_I.set_data(times, Ivals)
    line_R.set_data(times, Rvals)
    return im, line_S, line_I, line_R, time_text


ani = animation.FuncAnimation(fig, update_frame,
                              frames=len(frames_grid),
                              interval=1000 / fps,
                              blit=False)

print("Guardando animación como GIF...")
saved = False
try:
    gif_writer = animation.PillowWriter(fps=fps)
    ani.save(out_path_gif, writer=gif_writer, dpi=dpi)
    print("GIF guardado en:", out_path_gif)
    saved = True
except Exception as e:
    print("Error al guardar GIF:", e)


df = pd.DataFrame({"time": time_hist, "S": S_hist, "I": I_hist, "R": R_hist})
df.to_csv(out_csv, index=False)
print("Datos SIR guardados en:", out_csv)

plt.close(fig)
fig
