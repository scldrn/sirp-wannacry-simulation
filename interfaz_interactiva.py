#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interfaz interactiva para el modelo SIRP de WannaCry
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import ipywidgets as widgets
from IPython.display import display, HTML
from modelo_sirp import ModeloSIRP

class InterfazSIRP:
    def __init__(self):
        self.modelo = ModeloSIRP()
        self.modelo.simular_todos_escenarios()
        
    def crear_interfaz(self):
        """Crea una interfaz interactiva para explorar los resultados"""
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
                self.modelo.simular_escenario(escenario)
                
                # Mostrar visualización seleccionada
                if tipo_viz_widget.value == 'Curvas SIRP':
                    self.modelo.graficar_curvas([escenario])
                    plt.show()
                else:
                    # Para la animación, mostramos solo algunos frames estáticos
                    # ya que la animación completa es más adecuada para guardar como archivo
                    fig, ax = plt.subplots(figsize=(10, 8))
                    plt.title(f"Vista previa de animación para escenario: {escenario}")
                    plt.text(0.5, 0.5, "La animación completa se puede generar\ny guardar ejecutando el script principal",
                            ha='center', va='center', fontsize=14, transform=ax.transAxes)
                    plt.show()
                    
                    print("Para generar la animación completa, ejecute:")
                    print(f"modelo = ModeloSIRP()")
                    print(f"modelo.simular_escenario('{escenario}')")
                    print(f"ani, fig = modelo.animar_propagacion('{escenario}', guardar=True)")
        
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
        actualizar_btn.click()

# Función para iniciar la interfaz
def iniciar_interfaz():
    interfaz = InterfazSIRP()
    interfaz.crear_interfaz()
    return interfaz

if __name__ == "__main__":
    # Este script está diseñado para ejecutarse en un entorno Jupyter
    print("Este script está diseñado para ejecutarse en un entorno Jupyter/IPython.")
    print("Para usarlo, importe y ejecute la función iniciar_interfaz().")