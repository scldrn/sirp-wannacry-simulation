# Simulación de Propagación de WannaCry con Modelo SIRP (RK4)

Este proyecto implementa una simulación de la propagación del ransomware WannaCry
usando un modelo epidemiológico SIRP (Susceptibles, Infectados, Removidos, Protegidos).
La integración del sistema de ecuaciones diferenciales se realiza con
Runge–Kutta de 4º orden (RK4) implementado manualmente, sin depender de SciPy.

## Contenidos del repositorio

- `modelo_sirp_rk4.py`: implementación del modelo SIRP, simulación de escenarios,
  gráficas y animaciones de propagación en red.
- `interfaz_interactiva_rk4.py`: interfaz basada en `ipywidgets` para explorar
  escenarios y visualizar resultados dentro de Jupyter.
- `simulacion_wannacry_rk4.ipynb`: notebook que ejecuta la simulación completa,
  genera visualizaciones y lanza la interfaz interactiva.
- `animacion_*.mp4`: archivos de video con las animaciones por escenario.
- `executed_notebook.ipynb`: notebook ya ejecutado (salida generada por nbconvert).

## Modelo SIRP

- `S` (Susceptibles): equipos vulnerables al contagio.
- `I` (Infectados): equipos afectados por el ransomware.
- `R` (Removidos): equipos limpiados o que salen de la red.
- `P` (Protegidos): equipos parcheados o con mitigación activa.

Medidas y parámetros principales:
- `beta0`: tasa de contagio base.
- `kappa`: reducción de contagio tras activar el kill-switch en `t_k`.
- `gamma`: tasa de limpieza (I→R).
- `q`: tasa adicional de salida de infectados por cuarentena (I→R).
- `u_p`: tasa de parcheo (S→P) activa desde `t_parcheo`.
- `omega_P`: pérdida de protección (P→S).

Escenarios incluidos:
- `baseline`: sin mitigaciones.
- `kill_switch`: kill-switch que reduce contagio a partir de `t_k`.
- `parcheo`: parcheo progresivo desde `t_parcheo` a tasa `u_p`.
- `cuarentena`: cuarentena con tasa `q`.
- `combinado`: combinación de las medidas anteriores.

## Requisitos

Recomendado: Python 3.8+.

Dependencias Python:
- `numpy`, `matplotlib`, `networkx`, `tqdm`, `ipywidgets`, `nbconvert`.

Para exportar animaciones MP4:
- `ffmpeg` instalado y disponible en el PATH.

Instalación rápida de dependencias (ejemplos):

```bash
python -m pip install numpy matplotlib networkx tqdm ipywidgets nbconvert
# En macOS, instalar ffmpeg (ejemplo con Homebrew):
brew install ffmpeg
```

Habilitar ipywidgets en Jupyter Lab si fuese necesario:

```bash
jupyter nbextension enable --py widgetsnbextension
```

## Uso rápido

### Ejecutar el notebook

1. Abrir `simulacion_wannacry_rk4.ipynb` en Jupyter.
2. Ejecutar todas las celdas para simular escenarios, ver las curvas y abrir la interfaz.

Alternativa por CLI (ejecuta y guarda resultado):

```bash
jupyter nbconvert --to notebook --execute simulacion_wannacry_rk4.ipynb --output executed_notebook.ipynb
```

### Iniciar la interfaz interactiva

- En el notebook, ejecutar la celda con:

```python
from interfaz_interactiva_rk4 import iniciar_interfaz
interfaz = iniciar_interfaz()
```

La interfaz permite:
- Seleccionar escenario.
- Ajustar parámetros (`beta0`, `kappa`, `gamma`, `q`, `u_p`).
- Visualizar curvas SIRP o la animación de propagación en red.

### Generar animaciones MP4

Desde el notebook o script:

```python
from modelo_sirp_rk4 import ModeloSIRP
modelo = ModeloSIRP()
modelo.simular_todos_escenarios()
for escenario in modelo.escenarios:
    print(f"Creando animación para escenario: {escenario}")
    modelo.animar_propagacion(escenario, guardar=True)
```

Genera archivos `animacion_<escenario>.mp4` (requiere `ffmpeg`).

## Ejemplo de uso programático

```python
from modelo_sirp_rk4 import ModeloSIRP

modelo = ModeloSIRP()
modelo.simular_escenario('baseline')
fig, axs = modelo.graficar_curvas(['baseline'])
```

## Notas de implementación

- Integración RK4 manual con corrección numérica: se recorta a no negativos y
  se reescala para conservar aproximadamente la población total `N`.
- La animación usa un grafo Watts–Strogatz para visualización (no es la red real).
- Los docstrings en `modelo_sirp_rk4.py` y `interfaz_interactiva_rk4.py` explican
  parámetros, retornos y comportamiento de cada método.

## Estructura del proyecto

```
├── modelo_sirp_rk4.py
├── interfaz_interactiva_rk4.py
├── simulacion_wannacry_rk4.ipynb
├── executed_notebook.ipynb
└── animacion_*.mp4
```

## Solución de problemas

- Error de nbconvert sobre formato: asegúrate de que las celdas de código del notebook
  incluyen la clave `outputs` (el notebook provisto está corregido).
- Falta `ffmpeg`: instálalo y verifica que `ffmpeg -version` funciona en tu terminal.
- Widgets sin mostrarse: habilita `widgetsnbextension` o usa Jupyter clásico.

## Licencia

Este proyecto es para fines educativos y de simulación.
