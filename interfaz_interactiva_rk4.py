#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interfaz interactiva para el modelo SIRP de WannaCry.

Este módulo provee una interfaz basada en `ipywidgets` para explorar escenarios,
ajustar parámetros del modelo y visualizar tanto curvas SIRP como animaciones
de propagación sobre una red.

Uso principal: importar `iniciar_interfaz()` desde un notebook Jupyter y
ejecutar la celda para mostrar controles y salidas.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import ipywidgets as widgets
from IPython.display import display, HTML
from modelo_sirp_rk4 import ModeloSIRP

class InterfazSIRP:
    """Contiene el estado del modelo y los widgets de interacción.

    - Instancia internamente `ModeloSIRP` y simula todos los escenarios.
    - Ofrece `crear_interfaz()` para construir y mostrar los controles.
    """
    def __init__(self):
        self.modelo = ModeloSIRP()
        self.modelo.simular_todos_escenarios()
        
    def crear_interfaz(self):
        """Crea y muestra la interfaz con controles y salidas.

        Controles:
        - Selección de escenario y tipo de visualización.
        - Sliders para ajustar parámetros (`beta0`, `kappa`, `gamma`, `q`, `u_p`).

        Salida:
        - Curvas SIRP o animación de red según elección.
        """
        # Widget para seleccionar escenario
        escenario_widget = widgets.Dropdown(
            options=list(self.modelo.escenarios.keys()),
            value='baseline',
            description='Escenario:',
            style={'description_width': 'initial'}
        )
        
        # Widget para seleccionar tipo de visualización
        tipo_viz_widget = widgets.RadioButtons(
            options=['Curvas SIRP', 'Animación de Red'],
            value='Curvas SIRP',
            description='Visualización:',
            style={'description_width': 'initial'}
        )
        
        # Widget para ajustar parámetros
        beta0_widget = widgets.FloatSlider(
            value=self.modelo.beta0,
            min=0.1,
            max=1.0,
            step=0.01,
            description='β₀:',
            style={'description_width': 'initial'}
        )
        
        kappa_widget = widgets.FloatSlider(
            value=self.modelo.kappa,
            min=0.01,
            max=1.0,
            step=0.01,
            description='κ:',
            style={'description_width': 'initial'}
        )
        
        gamma_widget = widgets.FloatSlider(
            value=self.modelo.gamma,
            min=0.01,
            max=0.5,
            step=0.01,
            description='γ:',
            style={'description_width': 'initial'}
        )
        
        q_widget = widgets.FloatSlider(
            value=self.modelo.q,
            min=0.0,
            max=0.5,
            step=0.01,
            description='q:',
            style={'description_width': 'initial'}
        )
        
        u_p_widget = widgets.FloatSlider(
            value=self.modelo.u_p,
            min=0.0,
            max=1.0,
            step=0.01,
            description='u_p:',
            style={'description_width': 'initial'}
        )
        
        # Botón para actualizar simulación
        actualizar_btn = widgets.Button(
            description='Actualizar Simulación',
            button_style='primary',
            tooltip='Ejecutar simulación con los parámetros actuales'
        )
        
        # Área de salida para gráficos
        output = widgets.Output()
        
        # Función para actualizar la visualización
        def actualizar_viz(b):
            with output:
                output.clear_output(wait=True)
                
                # Actualizar parámetros del modelo
                self.modelo.beta0 = beta0_widget.value
                self.modelo.kappa = kappa_widget.value
                self.modelo.gamma = gamma_widget.value
                self.modelo.q = q_widget.value
                self.modelo.u_p = u_p_widget.value
                
                # Ejecutar simulación para el escenario seleccionado
                escenario = escenario_widget.value
                # Corregido: usar el método existente en ModeloSIRP
                self.modelo.simular_escenario(escenario)
                
                # Mostrar visualización seleccionada
                if tipo_viz_widget.value == 'Curvas SIRP':
                    self.modelo.graficar_curvas([escenario])
                    plt.show()
                else:
                    # Mostrar la animación en el notebook usando JSHTML
                    ani, fig = self.modelo.animar_propagacion(escenario)
                    plt.close(fig)
                    display(HTML(ani.to_jshtml()))
        
        # Conectar botón con función de actualización
        actualizar_btn.on_click(actualizar_viz)
        
        # Organizar widgets en la interfaz
        param_box = widgets.VBox([
            widgets.HTML("<h3>Parámetros del Modelo</h3>"),
            beta0_widget,
            kappa_widget,
            gamma_widget,
            q_widget,
            u_p_widget
        ])
        
        viz_box = widgets.VBox([
            widgets.HTML("<h3>Opciones de Visualización</h3>"),
            escenario_widget,
            tipo_viz_widget,
            actualizar_btn
        ])
        
        # Layout principal
        main_box = widgets.HBox([param_box, viz_box])
        
        # Mostrar interfaz
        display(widgets.VBox([
            widgets.HTML("<h2>Simulación de Propagación de WannaCry - Modelo SIRP</h2>"),
            main_box,
            output
        ]))
        
        # Mostrar visualización inicial
        actualizar_viz(None)

# Función para iniciar la interfaz
def iniciar_interfaz():
    """Inicializa y muestra la interfaz interactiva en Jupyter.

    Retorna: instancia de `InterfazSIRP`.
    """
    interfaz = InterfazSIRP()
    interfaz.crear_interfaz()
    return interfaz

if __name__ == "__main__":
    # Este script está diseñado para ejecutarse en un entorno Jupyter
    print("Este script está diseñado para ejecutarse en un entorno Jupyter/IPython.")
    print("Para usarlo, importe y ejecute la función iniciar_interfaz().")