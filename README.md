# Simulación de Propagación de WannaCry con Modelo SIRP

Este proyecto implementa un modelo matemático SIRP (Susceptibles-Infectados-Removidos-Protegidos) para simular la propagación del ransomware WannaCry en una red de computadoras, demostrando el impacto de diferentes estrategias de mitigación como kill-switch, parcheo y cuarentena.

## Descripción del Proyecto

El proyecto modela la propagación de WannaCry en una red LAN homogénea de 256 equipos utilizando ecuaciones diferenciales que representan los estados SIRP:

- **S (Susceptibles)**: Equipos vulnerables que pueden ser infectados
- **I (Infectados)**: Equipos activamente infectados que pueden propagar el malware
- **R (Removidos)**: Equipos que han sido limpiados y no pueden reinfectarse
- **P (Protegidos)**: Equipos que han sido parcheados y son inmunes a la infección

## Modelo Matemático

El modelo utiliza las siguientes ecuaciones diferenciales:

```
Incidencia: λ(t) = β(t) · I/N, con 
β(t) = β0 si t < t_k; y β(t) = κ·β0 si t ≥ t_k (con κ = 0.1, t_k = 5 min) 

Sistema SIRP (1er orden): 
dS/dt = –λ(t)·S – u_p·S + ω_P·P 
dI/dt = λ(t)·S – (γ + q)·I 
dR/dt = (γ + q)·I 
dP/dt = u_p·S – ω_P·P 
```

### Parámetros del Modelo

- **β0 = 0.36**: Tasa de contagio base (antes del kill-switch)
- **κ = 0.10**: Factor de reducción tras activar el kill-switch
- **t_k = 5**: Minuto de activación del kill-switch
- **γ = 0.12**: Tasa de limpieza
- **q = 0.06**: Tasa de cuarentena efectiva
- **u_p = 0.30**: Tasa de parcheo (cuando está activo)
- **ω_P = 0.0002**: Tasa de pérdida de protección

## Escenarios Simulados

El proyecto simula cinco escenarios diferentes:

1. **Baseline (sin control)**: Sin medidas de mitigación
2. **Kill-switch**: Reducción de la tasa de contagio al 10% en t=5 min
3. **Parcheo**: Aplicación de parches desde t=1 min
4. **Cuarentena**: Aislamiento de equipos infectados
5. **Combinado**: Kill-switch + parcheo + cuarentena

## Requisitos

```
numpy==1.24.3
scipy==1.10.1
matplotlib==3.7.1
networkx==3.1
tqdm==4.65.0
ipywidgets==8.0.6
```

## Estructura del Proyecto

- `modelo_sirp.py`: Implementación del modelo matemático SIRP
- `interfaz_interactiva.py`: Interfaz para explorar los resultados de forma interactiva
- `simulacion_wannacry.ipynb`: Notebook Jupyter para ejecutar las simulaciones
- `requirements.txt`: Dependencias del proyecto

## Uso

### Ejecución desde línea de comandos

```bash
# Instalar dependencias
pip install -r requirements.txt

# Ejecutar simulación
python modelo_sirp.py
```

### Uso del Notebook Jupyter

```bash
# Iniciar Jupyter Notebook
jupyter notebook simulacion_wannacry.ipynb
```

### Interfaz Interactiva

La interfaz interactiva permite:
- Seleccionar diferentes escenarios
- Ajustar parámetros del modelo
- Visualizar curvas SIRP y animaciones de propagación

## Resultados

El proyecto genera:

1. **Gráficas de curvas SIRP**: Evolución temporal de los estados S, I, R y P
2. **Animaciones de propagación**: Visualización de la propagación en la red
3. **Comparativas entre escenarios**: Análisis del impacto de las diferentes estrategias

## Análisis de Umbral

- **Número reproductivo básico**: R0 = β0/(γ+q) = 0.36/0.18 = 2.0
- **Protección crítica**: p_c = 1 − 1/R0 = 0.5 (proteger ≥ 50% del parque)

## Autor

Este proyecto fue desarrollado como parte de un estudio sobre modelado epidemiológico aplicado a la ciberseguridad.