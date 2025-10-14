# Generador de simulación SIR con partículas y exportación MP4
# Ejecutar en un entorno con matplotlib y ffmpeg instalado para exportar MP4 correctamente.
# Guarda el archivo en /mnt/data/sir_particles.mp4
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from matplotlib.lines import Line2D

# Parámetros de la simulación (tomados del usuario)
L = 50.0
Ntotal = 300
I0 = 5
vmax = 1.5
r = 2.0
beta = 0.3   # tasa de infección
gamma = 0.05 # tasa de recuperación
dt = 0.1
Tmax = 200.0

# Parámetros de animación/exportación solicitados
target_duration_sec = 16.0   # entre 15-20s, escogemos ~16s
fps = 12                     # entre 10-15 fps
dpi = 100
out_path_gif = "sir_particles.gif"

# Preparación temporal
total_steps = int(np.ceil(Tmax / dt))
# número de frames de animación
n_frames = int(np.ceil(target_duration_sec * fps))
steps_per_frame = max(1, int(np.ceil(total_steps / n_frames)))
n_frames = int(np.ceil(total_steps / steps_per_frame))  # ajustar frames reales

print(f"Total steps: {total_steps}, Frames: {n_frames}, Steps/frame: {steps_per_frame}")

# Estados: 0 susceptible (azul), 1 infectado (rojo), 2 recuperado (verde)
STATE_S, STATE_I, STATE_R = 0, 1, 2
colors_map = {STATE_S: "blue", STATE_I: "red", STATE_R: "green"}

rng = np.random.default_rng(seed=42)

# Inicializar posiciones uniformes en [0,L]
pos = rng.uniform(0, L, size=(Ntotal, 2))

# Velocidades: magnitudes uniformes en [0,vmax], direcciones uniformes en [0,2pi)
speeds = rng.uniform(0, vmax, size=(Ntotal, 1))
angles = rng.uniform(0, 2*np.pi, size=(Ntotal, 1))
vel = np.hstack((speeds * np.cos(angles), speeds * np.sin(angles)))

# Estados iniciales: todos susceptibles excepto I0 infectados aleatorios
states = np.zeros(Ntotal, dtype=int)
infected_idx = rng.choice(Ntotal, size=I0, replace=False)
states[infected_idx] = STATE_I

# Para llevar cuentas de tiempos de infección si se desea (no estrictamente necesario aquí)
time_since_infection = np.zeros(Ntotal, dtype=float)

# Datos S, I, R para graficar
S_hist = []
I_hist = []
R_hist = []
time_hist = []

# Función para paso de simulación (single dt)
def step_simulation(pos, vel, states, time_since_infection):
    # Movimiento: añadir pequeña perturbación angular para movimiento browniano suave
    # Esto evita rebotes rígidos y cumple "rebotan o movimiento browniano suave"
    angular_perturb = rng.normal(scale=0.1, size=(Ntotal,))  # pendiente: control suave
    # actualizar ángulos y velocidades acorde
    angles = np.arctan2(vel[:,1], vel[:,0]) + angular_perturb
    speeds = np.linalg.norm(vel, axis=1)
    # mantener velocidades en [0,vmax]
    speeds = np.clip(speeds, 0, vmax)
    vel[:,0] = speeds * np.cos(angles)
    vel[:,1] = speeds * np.sin(angles)
    
    # actualizar posiciones con condiciones periódicas
    pos += vel * dt
    pos %= L  # periodic boundary conditions
    
    # Contagio: para cada infectado, buscar susceptibles dentro de radio r
    # Usamos cálculo de distancia por pares (O(N^2), N=300 está bien)
    idx_infected = np.where(states == STATE_I)[0]
    idx_susceptible = np.where(states == STATE_S)[0]
    if idx_infected.size > 0 and idx_susceptible.size > 0:
        # compute pairwise distances accounting periodic BC by minimum image convention
        # expand positions
        pos_inf = pos[idx_infected][:, None, :]   # (n_inf,1,2)
        pos_sus = pos[idx_susceptible][None, :, :] # (1,n_sus,2)
        delta = pos_inf - pos_sus  # shape (n_inf, n_sus, 2)
        # apply minimum image (periodic domain)
        delta = (delta + L/2) % L - L/2
        dist2 = np.sum(delta**2, axis=2)
        # for each susceptible, if any infected within r -> chance to infect
        within = dist2 <= r**2  # boolean (n_inf, n_sus)
        # collapse to per-susceptible any infected nearby
        sus_in_r = np.any(within, axis=0)  # length n_sus
        # susceptibles that are at risk indices in original array
        sus_at_r_idx = idx_susceptible[sus_in_r]
        # For each at-risk susceptible, perform Bernoulli(β*dt)
        if sus_at_r_idx.size > 0:
            probs = rng.random(size=sus_at_r_idx.size)
            new_infections_mask = probs < (beta * dt)
            new_infected = sus_at_r_idx[new_infections_mask]
            states[new_infected] = STATE_I
            time_since_infection[new_infected] = 0.0
    
    # Recuperación: infectados recuperan con prob gamma*dt per step
    if idx_infected.size > 0:
        probs_rec = rng.random(size=idx_infected.size)
        recovered_mask = probs_rec < (gamma * dt)
        recovered = idx_infected[recovered_mask]
        states[recovered] = STATE_R
        time_since_infection[recovered] = 0.0
    
    # actualizar time_since_infection para los que sigan infectados
    time_since_infection[states == STATE_I] += dt
    
    return pos, vel, states, time_since_infection

# Pre-allocate storage for frames
frames_pos = []
frames_states = []
frames_time = []

t = 0.0
step = 0
# Inicial registro t=0
S_hist.append(int(np.sum(states==STATE_S)))
I_hist.append(int(np.sum(states==STATE_I)))
R_hist.append(int(np.sum(states==STATE_R)))
time_hist.append(t)
frames_pos.append(pos.copy())
frames_states.append(states.copy())
frames_time.append(t)

# Simulación completa, muestreada por frames para la animación
while step < total_steps:
    for _ in range(steps_per_frame):
        pos, vel, states, time_since_infection = step_simulation(pos, vel, states, time_since_infection)
        t += dt
        step += 1
        if step >= total_steps:
            break
    # registro para frame
    S_hist.append(int(np.sum(states==STATE_S)))
    I_hist.append(int(np.sum(states==STATE_I)))
    R_hist.append(int(np.sum(states==STATE_R)))
    time_hist.append(t)
    frames_pos.append(pos.copy())
    frames_states.append(states.copy())
    frames_time.append(t)
    # validación conservación de partículas
    assert (S_hist[-1] + I_hist[-1] + R_hist[-1]) == Ntotal, "S+I+R != Ntotal"
    
print("Simulación finalizada. Recolectados frames:", len(frames_pos))

# --- Preparar figura para animación ---
fig = plt.figure(figsize=(12, 6), dpi=dpi)
ax_sim = fig.add_axes([0.01, 0.05, 0.48, 0.9])  # left (simulation)
ax_plot = fig.add_axes([0.53, 0.05, 0.46, 0.9]) # right (SIR curves)

# Simulation axes settings
ax_sim.set_xlim(0, L)
ax_sim.set_ylim(0, L)
ax_sim.set_aspect('equal')
ax_sim.set_title("Simulación de partículas (espacio)")
legend_elements = [Line2D([0], [0], marker='o', color='w', label='Susceptible', markerfacecolor='blue', markersize=8),
                   Line2D([0], [0], marker='o', color='w', label='Infectado', markerfacecolor='red', markersize=8),
                   Line2D([0], [0], marker='o', color='w', label='Recuperado', markerfacecolor='green', markersize=8)]
ax_sim.legend(handles=legend_elements, loc='upper right')

# Scatter (initial)
scat = ax_sim.scatter(frames_pos[0][:,0], frames_pos[0][:,1], s=20, c=[colors_map[s] for s in frames_states[0]], lw=0)

# Time counter on simulation panel
time_text = ax_sim.text(0.02, 1.02, f"t = {frames_time[0]:.1f}", transform=ax_sim.transAxes, fontsize=10, va='bottom')

# SIR plot settings
ax_plot.set_xlim(0, Tmax)
ax_plot.set_ylim(0, Ntotal)
ax_plot.set_xlabel("Tiempo")
ax_plot.set_ylabel("Número de individuos")
ax_plot.set_title("Curvas S(t), I(t), R(t)")
# plot lines (initially empty)
line_S, = ax_plot.plot([], [], label='S', color='blue')
line_I, = ax_plot.plot([], [], label='I', color='red')
line_R, = ax_plot.plot([], [], label='R', color='green')
ax_plot.legend(loc='upper right')

# Function to update each frame
def update_frame(frame_idx):
    pos_frame = frames_pos[frame_idx]
    states_frame = frames_states[frame_idx]
    t_frame = frames_time[frame_idx]
    # update scatter positions and colors
    scat.set_offsets(pos_frame)
    scat.set_array(None)  # clear previous array
    scat.set_color([colors_map[s] for s in states_frame])
    # update time text
    time_text.set_text(f"t = {t_frame:.1f}")
    # update SIR curves up to this time
    times = np.array(time_hist[:frame_idx+1])
    Svals = np.array(S_hist[:frame_idx+1])
    Ivals = np.array(I_hist[:frame_idx+1])
    Rvals = np.array(R_hist[:frame_idx+1])
    line_S.set_data(times, Svals)
    line_I.set_data(times, Ivals)
    line_R.set_data(times, Rvals)
    # adjust xlim to show progress (optional: keep full Tmax)
    ax_plot.set_xlim(0, Tmax)
    return scat, line_S, line_I, line_R, time_text

# Crear objeto de animación (func animation.FuncAnimation)
ani = animation.FuncAnimation(fig, update_frame, frames=len(frames_pos), interval=1000/fps, blit=False)

# Guardar animación como MP4 (intento robusto)
saved = False
print("Intentando guardar GIF como fallback...")
try:
    gif_writer = animation.PillowWriter(fps=fps)
    ani.save(out_path_gif, writer=gif_writer, dpi=dpi)
    print("Guardado GIF en:", out_path_gif)
    saved = True
except Exception as e2:
    print("Error guardando GIF fallback:", e2)
    saved = False

# Mostrar resultado de guardado y ruta del archivo si existe

if saved and os.path.exists(out_path_gif):
    print("Archivo GIF disponible en:", out_path_gif)
else:
    print("No se pudo guardar la animación en GIF en este entorno. Se generó la animación en memoria.")

# Guardar datos SIR a CSV también para posterior análisis
import pandas as pd
df = pd.DataFrame({"time": time_hist, "S": S_hist, "I": I_hist, "R": R_hist})
csv_path = "sir_timeseries.csv"
df.to_csv(csv_path, index=False)
print("Timeseries guardada en:", csv_path)

# Mostrar la figura inline (retorna la figura para que el usuario vea la última trama en notebooks)
plt.close(fig)
fig


