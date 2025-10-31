#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modelo SIRP para la propagación de WannaCry.

Este módulo implementa un modelo SIRP (Susceptibles, Infectados, Removidos,
Protegidos) para estudiar la propagación del ransomware WannaCry en una red.
La integración del sistema de EDO se realiza con Runge–Kutta de 4º orden (RK4)
implementado manualmente, evitando dependencias de SciPy.

Contenido:
- Clase `ModeloSIRP` con parámetros, escenarios, simulación y visualización.
- Gráficas de curvas SIRP y animaciones sobre un grafo mundo-pequeño.

Estados:
- `S`: Susceptibles (vulnerables), `I`: Infectados, `R`: Removidos, `P`: Protegidos.

Escenarios:
- `baseline`, `kill_switch`, `parcheo`, `cuarentena`, `combinado`.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import networkx as nx
from tqdm import tqdm
import matplotlib.colors as mcolors  # Puede no usarse; se deja por compatibilidad


class ModeloSIRP:
    def __init__(self):
        # Parámetros del modelo
        #self.N = 256         # Número total de equipos
        #self.beta0 = 0.36    # Tasa de contagio base
        #self.kappa = 0.10    # Factor de reducción tras kill-switch
        #self.t_k = 5         # Tiempo de activación del kill-switch (minutos)
        #self.gamma = 0.12    # Tasa de limpieza
        #self.q = 0.06        # Tasa de cuarentena
        #self.u_p = 0.30      # Tasa de parcheo
        #self.omega_P = 0.0002  # Tasa de pérdida de protección

        self.N = 256
        self.beta0 = 0.45     # mayor contagio
        self.kappa = 0.15     # kill-switch menos drástico
        self.t_k = 8          # se activa un poco más tarde
        self.u_p = 0.15       # parcheo más gradual
        self.t_parcheo = 3    # aplica el parche a partir de t=3
        self.gamma = 0.1      # limpieza un poco más lenta
        self.q = 0.08         # cuarentena más efectiva
        self.omega_P = 0.0002  # tasa de pérdida de protección






        # Condiciones iniciales
        self.S0 = 255
        self.I0 = 10
        self.R0 = 0
        self.P0 = 0



        # Horizonte de tiempo
        self.t_span = (0, 60)                       # 60 minutos
        self.t_eval = np.linspace(0, 60, 601)       # Puntos de evaluación (cada 0.1 min)

        # Escenarios
        self.escenarios = {
            'baseline':   {'kill_switch': False, 'u_p': 0,     'q': 0},
            'kill_switch':{'kill_switch': True,  'u_p': 0,     'q': 0},
            'parcheo':    {'kill_switch': False, 'u_p': 0.30,  'q': 0,   't_parcheo': 1},
            'cuarentena': {'kill_switch': False, 'u_p': 0,     'q': 0.06},
            'combinado':  {'kill_switch': True,  'u_p': 0.30,  'q': 0.06, 't_parcheo': 1}
        }

        # Resultados
        self.resultados = {}

    # -----------------------------
    # Dinámica del modelo
    # -----------------------------
    def beta(self, t, kill_switch=True):
        """Devuelve la tasa de contagio efectiva en tiempo `t`.

        Si `kill_switch` es True y `t >= self.t_k`, aplica reducción `beta0 * kappa`;
        en caso contrario devuelve `beta0`.

        Parámetros:
        - `t`: tiempo (min).
        - `kill_switch`: bool, activa el cambio de tasa tras `t_k`.
        """
        if kill_switch and t >= self.t_k:
            return self.beta0 * self.kappa
        else:
            return self.beta0

    def modelo_sirp(self, t, y, kill_switch=True, u_p=0, q=0, t_parcheo=None):
        """Sistema SIRP en forma de derivadas.

        Parámetros:
        - `t`: tiempo actual.
        - `y`: estado `[S, I, R, P]`.
        - `kill_switch`: activa reducción de contagio tras `t_k`.
        - `u_p`: tasa de parcheo (S→P), activa tras `t_parcheo`.
        - `q`: tasa adicional de salida de infectados (cuarentena).
        - `t_parcheo`: inicio del parcheo; si None, siempre activo.

        Retorna: `[dSdt, dIdt, dRdt, dPdt]`.
        """
        S, I, R, P = y

        # Aplicar parcheo solo después de t_parcheo
        u_p_actual = u_p if (t_parcheo is None or t >= t_parcheo) else 0

        # Tasa de incidencia
        beta_t = self.beta(t, kill_switch)
        lambda_t = beta_t * I / self.N

        # Sistema de ecuaciones
        dSdt = -lambda_t * S - u_p_actual * S + self.omega_P * P
        dIdt =  lambda_t * S - (self.gamma + q) * I
        dRdt = (self.gamma + q) * I
        dPdt =  u_p_actual * S - self.omega_P * P

        return [dSdt, dIdt, dRdt, dPdt]

    # -----------------------------
    # Integrador RK4 manual
    # -----------------------------
    def _rhs_escenario(self, escenario):
        """Construye `f(t, y)` para el `escenario` dado.

        Extrae parámetros del diccionario y retorna una función que llama a
        `modelo_sirp` con dichos parámetros fijados.
        """
        kill_switch = escenario.get('kill_switch', False)
        u_p        = escenario.get('u_p', 0)
        q          = escenario.get('q', 0)
        t_parcheo  = escenario.get('t_parcheo', None)

        def f(t, y):
            return self.modelo_sirp(
                t, y,
                kill_switch=kill_switch,
                u_p=u_p,
                q=q,
                t_parcheo=t_parcheo
            )
        return f

    def _rk4_integrate(self, f, t_eval, y0):
        """Integra `y' = f(t, y)` mediante RK4 clásico.

        Parámetros:
        - `f`: callable `(t, y) -> derivadas`.
        - `t_eval`: tiempos crecientes donde se almacena la solución.
        - `y0`: estado inicial `[S0, I0, R0, P0]`.

        Retorna:
        - `Y`: array `(len(t_eval), 4)` con la trayectoria.

        Notas numéricas:
        - Se aplica `clip` para evitar negativos.
        - Se reescala para mantener suma ≈ `N`.
        """

        y = np.array(y0, dtype=float)
        Y = np.zeros((len(t_eval), len(y0)), dtype=float)
        Y[0] = y

        for i in range(1, len(t_eval)):
            t = t_eval[i - 1]
            h = t_eval[i] - t  # tamaño de paso

            # Clásico RK4
            k1 = np.array(f(t, y))
            k2 = np.array(f(t + 0.5*h, y + 0.5*h*k1))
            k3 = np.array(f(t + 0.5*h, y + 0.5*h*k2))
            k4 = np.array(f(t + h,     y + h*k3))

            y = y + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)

            # Correcciones numéricas: no negativos y conservación aproximada de la población
            y = np.clip(y, 0.0, self.N)
            total = y.sum()
            if total > 0:
                y *= (self.N / total)

            Y[i] = y

        return Y

    # -----------------------------
    # Simulación por escenarios
    # -----------------------------
    def simular_escenario(self, nombre_escenario):
        """Simula un escenario específico y guarda resultados.

        Parámetros:
        - `nombre_escenario`: clave de `self.escenarios`.

        Retorna: diccionario con `tiempo`, `S`, `I`, `R`, `P`.
        """
        escenario = self.escenarios[nombre_escenario]

        # Condiciones iniciales
        y0 = [self.S0, self.I0, self.R0, self.P0]

        # Construir f(t,y) con parámetros del escenario e integrar
        f = self._rhs_escenario(escenario)
        Y = self._rk4_integrate(f, self.t_eval, y0)

        # Guardar resultados
        self.resultados[nombre_escenario] = {
            'tiempo': self.t_eval,
            'S': Y[:, 0],
            'I': Y[:, 1],
            'R': Y[:, 2],
            'P': Y[:, 3]
        }

        return self.resultados[nombre_escenario]

    def simular_todos_escenarios(self):
        """Itera y simula todos los escenarios definidos en `self.escenarios`."""
        for nombre in tqdm(self.escenarios.keys(), desc="Simulando escenarios"):
            self.simular_escenario(nombre)

    # -----------------------------
    # Visualización
    # -----------------------------
    def graficar_curvas(self, escenarios=None, guardar=False):
        """Grafica las curvas S, I, R, P para los escenarios.

        Parámetros:
        - `escenarios`: lista de escenarios; si None usa todos.
        - `guardar`: si True, guarda `curvas_sirp.png`.

        Retorna: `(fig, axs)`.
        """
        if escenarios is None:
            escenarios = list(self.escenarios.keys())

        fig, axs = plt.subplots(len(escenarios), 1, figsize=(10, 3*len(escenarios)), sharex=True)
        if len(escenarios) == 1:
            axs = [axs]

        for i, nombre in enumerate(escenarios):
            if nombre not in self.resultados:
                continue

            res = self.resultados[nombre]
            ax = axs[i]

            ax.plot(res['tiempo'], res['S'], 'b-', label='Susceptibles')
            ax.plot(res['tiempo'], res['I'], 'r-', label='Infectados')
            ax.plot(res['tiempo'], res['R'], 'g-', label='Removidos')
            ax.plot(res['tiempo'], res['P'], 'y-', label='Protegidos')

            # Marcar eventos importantes
            if self.escenarios[nombre].get('kill_switch', False):
                ax.axvline(x=self.t_k, color='k', linestyle='--', alpha=0.5, label='Kill-Switch')

            if 't_parcheo' in self.escenarios[nombre]:
                ax.axvline(x=self.escenarios[nombre]['t_parcheo'], color='purple',
                           linestyle='--', alpha=0.5, label='Inicio Parcheo')

            ax.set_title(f'Escenario: {nombre}')
            ax.set_ylabel('Número de equipos')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right')

        axs[-1].set_xlabel('Tiempo (minutos)')
        plt.tight_layout()

        if guardar:
            plt.savefig('curvas_sirp.png', dpi=300, bbox_inches='tight')

        return fig, axs

    # -----------------------------
    # Red y animación (NetworkX)
    # -----------------------------
    def crear_grafo_red(self):
        """Crea un grafo tipo mundo-pequeño (Watts–Strogatz) para visualización."""
        # Crear un grafo de Watts-Strogatz (mundo pequeño) para visualización
        G = nx.watts_strogatz_graph(self.N, k=8, p=0.1)
        return G

    def animar_propagacion(self, escenario='baseline', guardar=False):
        """Genera una animación de la propagación en la red.

        Parámetros:
        - `escenario`: nombre del escenario a animar.
        - `guardar`: si True, exporta `animacion_<escenario>.mp4` (requiere ffmpeg).

        Retorna: `(ani, fig)` (`FuncAnimation`, figura base).
        """
        if escenario not in self.resultados:
            self.simular_escenario(escenario)

        # Crear grafo
        G = self.crear_grafo_red()
        pos = nx.spring_layout(G, seed=42)  # Posiciones fijas para los nodos

        # Preparar datos para la animación
        res = self.resultados[escenario]
        tiempo = res['tiempo']
        S = res['S']
        I = res['I']
        R = res['R']
        P = res['P']

        # Crear figura
        fig, ax = plt.subplots(figsize=(10, 8))

        # Colores para cada estado
        colores = {'S': 'blue', 'I': 'red', 'R': 'green', 'P': 'yellow'}

        # Función para actualizar la animación
        def update(frame):
            ax.clear()

            # Calcular la proporción de cada estado en este frame
            total = self.N
            prop_S = S[frame] / total
            prop_I = I[frame] / total
            prop_R = R[frame] / total
            prop_P = P[frame] / total

            # Asignar estados a los nodos basados en las proporciones
            estados = []
            for _ in range(int(prop_S * self.N)):
                estados.append('S')
            for _ in range(int(prop_I * self.N)):
                estados.append('I')
            for _ in range(int(prop_R * self.N)):
                estados.append('R')
            for _ in range(int(prop_P * self.N)):
                estados.append('P')

            # Ajustar para asegurar exactamente N estados
            while len(estados) < self.N:
                estados.append('S')
            while len(estados) > self.N:
                estados.pop()

            # Mezclar aleatoriamente los estados
            np.random.shuffle(estados)

            # Asignar colores a los nodos
            node_colors = [colores[estado] for estado in estados]

            # Dibujar el grafo
            nx.draw(G, pos, node_color=node_colors, with_labels=False,
                    node_size=50, edge_color='gray', alpha=0.8, ax=ax)

            # Añadir leyenda
            for estado, color in colores.items():
                ax.plot([], [], 'o', color=color, label=estado)

            # Añadir información del tiempo
            ax.set_title(f'Propagación de WannaCry - Escenario: {escenario}\nTiempo: {tiempo[frame]:.1f} minutos')

            # Añadir leyenda de estados con conteos enteros
            labels = {
                'S': f'Susceptibles: {int(S[frame])}',
                'I': f'Infectados: {int(I[frame])}',
                'R': f'Removidos: {int(R[frame])}',
                'P': f'Protegidos: {int(P[frame])}'
            }

            handles = [plt.Line2D([0], [0], marker='o', color='w',
                                  markerfacecolor=colores[estado], markersize=10, label=labels[estado])
                       for estado in ['S', 'I', 'R', 'P']]

            ax.legend(handles=handles, loc='upper right')

            # Marcar eventos importantes
            if self.escenarios[escenario].get('kill_switch', False) and tiempo[frame] >= self.t_k:
                ax.text(0.02, 0.02, 'Kill-Switch Activado', transform=ax.transAxes,
                        color='red', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))

            if 't_parcheo' in self.escenarios[escenario] and tiempo[frame] >= self.escenarios[escenario]['t_parcheo']:
                ax.text(0.02, 0.06, 'Parcheo Activo', transform=ax.transAxes,
                        color='blue', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))

            if self.escenarios[escenario].get('q', 0) > 0:
                ax.text(0.02, 0.10, 'Cuarentena Activa', transform=ax.transAxes,
                        color='green', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))

        # Crear animación (usar menos frames para mejor rendimiento)
        frames = list(range(0, len(tiempo), 10))  # Tomar un frame cada 10 para la animación
        ani = FuncAnimation(fig, update, frames=frames, interval=100, blit=False)

        if guardar:
            ani.save(f'animacion_{escenario}.mp4', writer='ffmpeg', fps=10, dpi=200)

        return ani, fig


# Función para ejecutar la simulación completa
def ejecutar_simulacion():
    """Ejecuta la simulación completa, guarda curvas y animaciones.

    - Simula todos los escenarios.
    - Guarda las curvas SIRP (`curvas_sirp.png`).
    - Genera animaciones MP4 para cada escenario.

    Retorna: instancia `ModeloSIRP` con resultados.
    """
    modelo = ModeloSIRP()
    modelo.simular_todos_escenarios()

    # Graficar curvas para todos los escenarios
    modelo.graficar_curvas(guardar=True)

    # Crear animaciones para cada escenario
    for escenario in modelo.escenarios:
        print(f"Creando animación para escenario: {escenario}")
        modelo.animar_propagacion(escenario, guardar=True)

    return modelo


if __name__ == "__main__":
    modelo = ejecutar_simulacion()
