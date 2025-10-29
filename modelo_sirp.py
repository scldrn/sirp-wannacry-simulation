#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modelo SIRP para la propagación de WannaCry
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import networkx as nx
from tqdm import tqdm
import matplotlib.colors as mcolors

class ModeloSIRP:
    def __init__(self):
        # Parámetros del modelo
        self.N = 256  # Número total de equipos
        self.beta0 = 0.36  # Tasa de contagio base
        self.kappa = 0.10  # Factor de reducción tras kill-switch
        self.t_k = 5  # Tiempo de activación del kill-switch (minutos)
        self.gamma = 0.12  # Tasa de limpieza
        self.q = 0.06  # Tasa de cuarentena
        self.u_p = 0.30  # Tasa de parcheo
        self.omega_P = 0.0002  # Tasa de pérdida de protección
        
        # Condiciones iniciales
        self.S0 = 255
        self.I0 = 1
        self.R0 = 0
        self.P0 = 0
        
        # Horizonte de tiempo
        self.t_span = (0, 60)  # 60 minutos
        self.t_eval = np.linspace(0, 60, 601)  # Puntos de evaluación (cada 0.1 min)
        
        # Escenarios
        self.escenarios = {
            'baseline': {'kill_switch': False, 'u_p': 0, 'q': 0},
            'kill_switch': {'kill_switch': True, 'u_p': 0, 'q': 0},
            'parcheo': {'kill_switch': False, 'u_p': 0.30, 'q': 0, 't_parcheo': 1},
            'cuarentena': {'kill_switch': False, 'u_p': 0, 'q': 0.06},
            'combinado': {'kill_switch': True, 'u_p': 0.30, 'q': 0.06, 't_parcheo': 1}
        }
        
        # Resultados
        self.resultados = {}
        
    def beta(self, t, kill_switch=True):
        """Función para la tasa de contagio variable con kill-switch"""
        if kill_switch and t >= self.t_k:
            return self.beta0 * self.kappa
        else:
            return self.beta0
    
    def modelo_sirp(self, t, y, kill_switch=True, u_p=0, q=0, t_parcheo=None):
        """Sistema de ecuaciones diferenciales del modelo SIRP"""
        S, I, R, P = y
        
        # Aplicar parcheo solo después de t_parcheo
        u_p_actual = u_p if (t_parcheo is None or t >= t_parcheo) else 0
        
        # Tasa de incidencia
        beta_t = self.beta(t, kill_switch)
        lambda_t = beta_t * I / self.N
        
        # Sistema de ecuaciones
        dSdt = -lambda_t * S - u_p_actual * S + self.omega_P * P
        dIdt = lambda_t * S - (self.gamma + q) * I
        dRdt = (self.gamma + q) * I
        dPdt = u_p_actual * S - self.omega_P * P
        
        return [dSdt, dIdt, dRdt, dPdt]
    
    def simular_escenario(self, nombre_escenario):
        """Simula un escenario específico"""
        escenario = self.escenarios[nombre_escenario]
        
        # Condiciones iniciales
        y0 = [self.S0, self.I0, self.R0, self.P0]
        
        # Resolver el sistema de ecuaciones
        sol = solve_ivp(
            lambda t, y: self.modelo_sirp(
                t, y, 
                kill_switch=escenario.get('kill_switch', False),
                u_p=escenario.get('u_p', 0),
                q=escenario.get('q', 0),
                t_parcheo=escenario.get('t_parcheo', None)
            ),
            self.t_span,
            y0,
            t_eval=self.t_eval,
            method='RK45'
        )
        
        self.resultados[nombre_escenario] = {
            'tiempo': sol.t,
            'S': sol.y[0],
            'I': sol.y[1],
            'R': sol.y[2],
            'P': sol.y[3]
        }
        
        return sol
    
    def simular_todos_escenarios(self):
        """Simula todos los escenarios definidos"""
        for nombre in tqdm(self.escenarios.keys(), desc="Simulando escenarios"):
            self.simular_escenario(nombre)
    
    def graficar_curvas(self, escenarios=None, guardar=False):
        """Grafica las curvas SIRP para los escenarios especificados"""
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
            plt.savefig(f'curvas_sirp.png', dpi=300, bbox_inches='tight')
        
        return fig, axs
    
    def crear_grafo_red(self):
        """Crea un grafo para representar la red de equipos"""
        # Crear un grafo completo para representar la red homogénea
        G = nx.Graph()
        
        # Añadir nodos (equipos)
        for i in range(self.N):
            G.add_node(i)
        
        # Añadir conexiones (simplificado para visualización)
        # En lugar de un grafo completo (demasiado denso), creamos una red más visual
        # Usamos un modelo de Watts-Strogatz para una red de mundo pequeño
        G = nx.watts_strogatz_graph(self.N, k=8, p=0.1)
        
        return G
    
    def animar_propagacion(self, escenario='baseline', guardar=False):
        """Crea una animación de la propagación en la red"""
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
            
            # Ajustar para asegurar que tenemos exactamente N estados
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
            
            # Añadir leyenda de estados
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