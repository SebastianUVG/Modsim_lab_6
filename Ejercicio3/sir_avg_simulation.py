import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys, os

# === Rutas para importar tus módulos ===
sys.path.append(os.path.join(os.path.dirname(__file__), "../Ejercicio1"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../Ejercicio2"))

from sir_simulation import step_simulation   # función de partículas
from sir_ac_simulation import step, count_infected_neighbors  # funciones del autómata

Nexp = 10
Tmax = 150
seed = 42


def run_particle_model():
    """Ejecuta una simulación de partículas SIR"""
    from sir_simulation import L, Ntotal, beta, gamma, vmax, r, dt
    rng = np.random.default_rng(seed)

    # Estado inicial fijo
    pos = rng.uniform(0, L, size=(Ntotal, 2))
    speeds = rng.uniform(0, vmax, size=(Ntotal, 1))
    angles = rng.uniform(0, 2 * np.pi, size=(Ntotal, 1))
    vel = np.hstack((speeds * np.cos(angles), speeds * np.sin(angles)))

    STATE_S, STATE_I, STATE_R = 0, 1, 2
    states = np.zeros(Ntotal, dtype=int)
    infected_idx = rng.choice(Ntotal, size=5, replace=False)
    states[infected_idx] = STATE_I
    time_since_infection = np.zeros(Ntotal)

    S_hist, I_hist, R_hist = [], [], []
    total_steps = Tmax
    for t in range(total_steps):
        S_hist.append(np.sum(states == STATE_S))
        I_hist.append(np.sum(states == STATE_I))
        R_hist.append(np.sum(states == STATE_R))
        pos, vel, states, time_since_infection = step_simulation(
            pos, vel, states, time_since_infection
        )
    return np.array(S_hist), np.array(I_hist), np.array(R_hist)


print("=== Simulando promedio de modelo de Partículas ===")
all_Sp, all_Ip, all_Rp = [], [], []
for exp in range(Nexp):
    np.random.seed(exp)
    S, I, R = run_particle_model()
    all_Sp.append(S)
    all_Ip.append(I)
    all_Rp.append(R)
    print(f"  → Simulación Partículas {exp+1}/{Nexp} completada")

S_mean_p = np.mean(all_Sp, axis=0)
I_mean_p = np.mean(all_Ip, axis=0)
R_mean_p = np.mean(all_Rp, axis=0)

pd.DataFrame({
    "time": np.arange(Tmax),
    "S_mean": S_mean_p, "I_mean": I_mean_p, "R_mean": R_mean_p
}).to_csv("sir_avg_particles.csv", index=False)
print(" Guardado: sir_avg_particles.csv")


M, N = 60, 60
I0 = 15
beta_c = 0.3
gamma_c = 0.05
r_c = 1
STATE_S, STATE_I, STATE_R = 0, 1, 2

def run_cellular_model(initial_grid):
    grid = initial_grid.copy()
    S_hist, I_hist, R_hist = [], [], []
    for t in range(Tmax):
        S_hist.append(np.sum(grid == STATE_S))
        I_hist.append(np.sum(grid == STATE_I))
        R_hist.append(np.sum(grid == STATE_R))
        grid = step(grid)
    return np.array(S_hist), np.array(I_hist), np.array(R_hist)

rng = np.random.default_rng(seed)
fixed_infected_idx = rng.choice(M * N, size=I0, replace=False)
base_grid = np.zeros((M, N), dtype=int)
for idx in fixed_infected_idx:
    base_grid[idx // N, idx % N] = STATE_I

print("\n=== Simulando promedio de Autómata Celular ===")
all_Sc, all_Ic, all_Rc = [], [], []
for exp in range(Nexp):
    np.random.seed(exp)
    S, I, R = run_cellular_model(base_grid)
    all_Sc.append(S)
    all_Ic.append(I)
    all_Rc.append(R)
    print(f"  → Simulación Celular {exp+1}/{Nexp} completada")

S_mean_c = np.mean(all_Sc, axis=0)
I_mean_c = np.mean(all_Ic, axis=0)
R_mean_c = np.mean(all_Rc, axis=0)

pd.DataFrame({
    "time": np.arange(Tmax),
    "S_mean": S_mean_c, "I_mean": I_mean_c, "R_mean": R_mean_c
}).to_csv("sir_avg_cellular.csv", index=False)
print("Guardado: sir_avg_cellular.csv")


plt.figure(figsize=(10,6))
plt.plot(S_mean_p, '--', color='blue', alpha=0.6, label="S Partículas")
plt.plot(I_mean_p, '--', color='red', alpha=0.6, label="I Partículas")
plt.plot(R_mean_p, '--', color='green', alpha=0.6, label="R Partículas")

plt.plot(S_mean_c, '-', color='blue', label="S Celular")
plt.plot(I_mean_c, '-', color='red', label="I Celular")
plt.plot(R_mean_c, '-', color='green', label="R Celular")

plt.title(f"Promedio de {Nexp} simulaciones — Comparación de métodos SIR")
plt.xlabel("Tiempo (pasos)")
plt.ylabel("Número promedio de individuos/celdas")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
