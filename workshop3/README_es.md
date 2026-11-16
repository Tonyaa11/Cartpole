<!-- VERSIÓN EN ESPAÑOL (solo para lectura del equipo). La entrega oficial es README.md en inglés. -->

# Taller 3: Aprendizaje por Refuerzo para Evitar Fallos en CartPole

## 1. Descripción General

Esta carpeta contiene el proyecto del Taller 3 del curso de Machine Learning (Semestre 2026-III, Universidad Distrital Francisco José de Caldas, Grupo 5). Los Talleres 1 y 2 *predecían* si un episodio de CartPole estaba a punto de fallar; aquí un agente *controla* el mismo sistema para evitar esos fallos. El proyecto incluye la formulación del MDP, un entorno propio de Gymnasium validado contra CartPole-v1, una Deep Q-Network implementada desde cero en PyTorch, una búsqueda de hiperparámetros (6 configuraciones x 3 semillas), cinco políticas de línea base (una de ellas es el clasificador del Taller 1 usado como política), evaluación de prueba en estados iniciales reservados y registro con TensorBoard.

**Autores:** Miguel Angel Torres Ramirez, Cristian David Romero Gil, Omar Esteban Peña Sanchez

| Componente del MDP | Definición |
|---|---|
| Estado | `(x, x_dot, theta, theta_dot)`, 4 valores continuos |
| Acciones | `0` = empujar a la izquierda (-10 N), `1` = empujar a la derecha (+10 N) |
| Transición | ecuaciones de movimiento del carro-péndulo, paso de Euler de 0.02 s (idéntico a CartPole-v1) |
| Recompensa | +1 por paso sobrevivido + shaping opcional basado en potenciales, a partir del hallazgo del Taller 1 (distancia a los límites de fallo) |
| Terminación | fallo si \|theta\| > 12 grados o \|x\| > 2.4 m; truncamiento a los 500 pasos |

Entregables que pide el taller y dónde están:

| Requerido | Ubicación |
|---|---|
| `src/env.py` | `workshop3/src/env.py` |
| `src/agent.py` | `workshop3/src/agent.py` |
| `src/baselines.py` | `workshop3/src/baselines.py` |
| `src/train.py` | `workshop3/src/train.py` |
| `src/evaluate.py` | `workshop3/src/evaluate.py` |
| `checkpoints/agent_best.pt` | `checkpoints/agent_best.pt` |
| `runs/workshop3/` | `runs/workshop3/` |
| Informe | `workshop3/report/main.tex` -> `group5_workshop3.pdf` |

El código del Taller 3 está en `workshop3/` porque la raíz del repositorio ya tiene los archivos del Taller 1 (por ejemplo `src/baselines.py`).

---

## 2. Requisitos

### Sistema

* Linux (se recomienda Ubuntu 20.04+), macOS, Windows o Google Colab
* Python 3.10+ (desarrollado con Python 3.13; en Colab funciona sin cambios)
* Solo CPU: la red Q tiene unos 17 mil parámetros, así que una GPU no acelera nada

### Dependencias

Instala los paquetes necesarios:

```bash
pip install torch
pip install "gymnasium>=1.0"
pip install tensorboard
pip install numpy
pip install pandas
pip install scikit-learn
pip install matplotlib
```

---

O usa el archivo requirements.txt

```bash
pip install -r workshop3/requirements.txt
```

---

## 3. Configuración del Entorno (Linux)

Crea y activa un entorno virtual:

```bash
python3 -m venv rl_env
source rl_env/bin/activate
```

Actualiza pip:

```bash
pip install --upgrade pip
```

Instala las dependencias (ver la sección anterior).

### Google Colab

Abre `workshop3/notebooks/workshop3_colab.ipynb` en Colab (File -> Open notebook -> GitHub) y ejecuta todas las celdas. Clona el repositorio, ejecuta en orden todos los pasos de abajo y descarga los resultados.

---

## 4. Estructura del Proyecto

```text
.
├── src/                          # Workshop 1 (reused: dataset.py, features.py, baselines.py)
├── data/                         # Workshop 1 dataset and split (used by the W1 classifier)
├── checkpoints/
│   ├── agent_best.pt             # best DQN agent (required deliverable)
│   └── workshop3_sweep/          # best checkpoint of every configuration x seed
├── runs/
│   └── workshop3/                # TensorBoard logs, CSV logs, JSON results
└── workshop3/
    ├── configs/
    │   └── sweep.json            # the 6 configurations of the hyperparameter search
    ├── src/
    │   ├── env.py                # custom Gymnasium environment + validation
    │   ├── agent.py              # DQN agent (Q-network, replay buffer, target network)
    │   ├── baselines.py          # random, alternating, angle rule, PD rule, W1-guided
    │   ├── train.py              # training loop, validation, sweep and selection
    │   ├── evaluate.py           # test evaluation (nominal and noisy)
    │   ├── w1_bridge.py          # rebuilds the Workshop 1 classifier
    │   ├── common.py             # paths and seeds shared by all scripts
    │   └── make_report_assets.py # figures and LaTeX tables of the report
    ├── notebooks/
    │   └── workshop3_colab.ipynb # step-by-step Colab runner
    ├── report/
    │   ├── main.tex              # IEEE report (Overleaf)
    │   ├── figures/              # generated figures
    │   └── generated/            # generated numbers and table rows
    ├── requirements.txt
    └── README.md
```

---

## 5. Ejecución de los Experimentos

Ejecuta todos los comandos desde la raíz del repositorio.

### 5.1 Validar el Entorno

```bash
python workshop3/src/env.py
```

Esto hará lo siguiente:

* Revisa la API de Gymnasium con `check_env`
* Compara la dinámica con CartPole-v1 de Gymnasium (las mismas semillas y acciones deben dar los mismos estados)
* Ejecuta 100 episodios con política aleatoria e imprime el retorno medio, la distribución de longitudes y la frecuencia de acciones
* Guarda el diagnóstico en `runs/workshop3/env_validation.json`

---

### 5.2 Revisar el Clasificador del Taller 1 (opcional)

```bash
python workshop3/src/w1_bridge.py
```

Reconstruye el modelo Gradient Boosting del Taller 1 (misma partición, características, escalador y umbral) y verifica que el código rápido de características usado dentro del ciclo de control coincide con `src/features.py`.

---

### 5.3 Entrenar una Sola Configuración

```bash
python workshop3/src/train.py --config reference --seed 42
```

---

### 5.4 Búsqueda de Hiperparámetros

```bash
python workshop3/src/train.py --sweep
```

Esto hará lo siguiente:

* Entrena las 6 configuraciones de `workshop3/configs/sweep.json` con las semillas 42, 7 y 99 (300 episodios cada una)
* Evalúa al agente voraz cada 10 episodios en 10 estados iniciales de validación y guarda el mejor checkpoint de cada corrida en `checkpoints/workshop3_sweep/`
* Selecciona la mejor configuración solo con resultados de validación y copia su mejor corrida a `checkpoints/agent_best.pt`
* Escribe `runs/workshop3/sweep_results.json`

Para rehacer solo la selección a partir de los registros guardados: `python workshop3/src/train.py --select-only`.

---

### 5.5 Evaluación de Prueba

```bash
python workshop3/src/evaluate.py
```

Esto hará lo siguiente:

* Evalúa al agente y a las cinco líneas base en 100 estados iniciales reservados, en el entorno nominal y con ruido en el actuador (2 N)
* Reporta retorno, tasa de éxito, causas de fallo, tasa de peligro W1 y |theta| medio
* Escribe `runs/workshop3/test_results.json` y `runs/workshop3/trajectories.json`

---

### 5.6 Figuras y Tablas de LaTeX

```bash
python workshop3/src/make_report_assets.py
```

Escribe las figuras en `workshop3/report/figures/` y todos los números del informe en `workshop3/report/generated/`. Sube toda la carpeta `workshop3/report/` a Overleaf y compila `main.tex`.

---

## 6. Visualización en TensorBoard

Para visualizar los registros de entrenamiento:

```bash
tensorboard --logdir runs/workshop3 --port 6006
```

Luego abre en el navegador:

```text
http://localhost:6006
```

En Colab: `%load_ext tensorboard` and `%tensorboard --logdir runs/workshop3`.

Los registros se organizan así:

```text
runs/workshop3/
  ├── reference/
  │   ├── seed_42/   # TensorBoard events + episodes.csv + validation.csv + summary.json
  │   ├── seed_7/
  │   └── seed_99/
  ├── lr_1e-4/
  ...
```

Se registra por episodio: retorno de la tarea, retorno con shaping, longitud del episodio, epsilon y pérdida TD; por validación: retorno de validación, retorno suavizado y cómo terminaron los episodios de validación.

---

## 7. Reproducibilidad

Todos los experimentos son reproducibles gracias a:

* Semillas fijas: corridas de entrenamiento 42, 7 y 99; estados iniciales de validación 10000-10009; de prueba 20000-20099 (tres conjuntos disjuntos)
* Configuraciones definidas en `workshop3/configs/sweep.json`
* Un presupuesto de entrenamiento fijo (300 episodios por corrida)
* Selección de la configuración solo con validación; el conjunto de prueba solo lo usa `evaluate.py`
* Todos los números y figuras del informe se generan a partir de los registros con `make_report_assets.py`

Para reproducir los resultados:

1. Configura el entorno
2. Ejecuta en orden las secciones 5.1, 5.4, 5.5 y 5.6
3. Visualiza las corridas en TensorBoard

El entrenamiento de redes neuronales puede cambiar con las versiones de las librerías y el hardware, así que los números del informe vienen de las corridas cuyos registros y checkpoints están en `runs/workshop3/` y `checkpoints/`. Ejecutar `train.py --select-only`, `evaluate.py` y `make_report_assets.py` sobre esas corridas reproduce todos los números del informe; volver a entrenar produce corridas nuevas cuyos números pueden ser distintos.

---

## 8. Artefactos de Registro

El directorio `runs/workshop3/` contiene:

* Eventos de TensorBoard y registros CSV de las 18 corridas de entrenamiento
* `sweep_results.json`: resúmenes por corrida, tabla por configuración y la selección
* `test_results.json`: métricas de prueba de cada política en ambas condiciones
* `trajectories.json`: episodios de prueba de ejemplo del agente seleccionado
* `env_validation.json`: diagnóstico de la política aleatoria

Se pueden consultar sin volver a entrenar.

---

## 9. Notas

* La búsqueda completa tarda de 20 a 40 minutos en CPU; la evaluación de prueba unos 3 minutos (la política guiada por W1 es la más lenta).
* DQN se degrada después de alcanzar una buena política en todas las configuraciones; el agente que se usa es el mejor checkpoint de validación de cada corrida, no el último.
* Una regla PD simple (`theta + 0.5 * theta_dot`) ya resuelve la tarea nominal: se reporta como línea base a propósito.
